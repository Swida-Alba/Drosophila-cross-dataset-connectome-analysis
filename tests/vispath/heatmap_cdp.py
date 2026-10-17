"""Headless Chrome over CDP, for tests that must *run* an exported page.

A string assertion on an emitted template cannot see what plotly does with the
keys the page hands it: the interactive heatmap's "Hide/Show Labels" button
passed every load-time check while throwing inside its click handler and
freezing the tick labels (plan `_plan/plan-2026-10-13-heatmap-label-toggle-dead.md`).
So the heatmap tests drive a real browser, for the same reason
`tests/core/test_visualize_skeleton_camera_shadow.py` does — "a DOM stub cannot
see a camera that only disagrees after a gesture".

Kept deliberately small: launch one page, evaluate JS, wait for a predicate.
Import from a test with

    sys.path.insert(0, str(Path(__file__).resolve().parent))   # tests/vispath
    from heatmap_cdp import Page, REQUIRES_CDP

(the vispath test dir is not a package, so the path insert is the import).
"""

import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

CHROME_CANDIDATES = (
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta',
)


def _find_chrome():
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ('google-chrome', 'chromium', 'chromium-browser', 'chrome'):
        found = shutil.which(name)
        if found:
            return found
    return None


CHROME = _find_chrome()

try:
    import websocket  # noqa: F401  (websocket-client)
    HAVE_WEBSOCKET = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_WEBSOCKET = False

#: Apply to any test that needs a live page. Skipping is never a pass: a host
#: without Chrome or a websocket client reports SKIPPED, and the plan's gate
#: (T4) is only satisfied where these tests actually run.
REQUIRES_CDP = pytest.mark.skipif(
    CHROME is None or not HAVE_WEBSOCKET,
    reason='needs a Chrome/Chromium binary and the websocket client for CDP')

# Collect what the page throws on its own, including inside click handlers,
# where an exception surfaces nowhere to the reader.
ARM_ERROR_WATCH = (
    "window.__probeErrors=[];"
    "window.addEventListener('error',function(e){"
    "window.__probeErrors.push(String(e.message));});"
    "window.addEventListener('unhandledrejection',function(e){"
    "window.__probeErrors.push('rejection: '+String(e.reason));});"
    "return true;")

#: A plotly page has data once its first trace carries its axis arrays.
PAGE_READY = ("var g=document.querySelector('.js-plotly-plot');"
              "return !!(g && g._fullLayout && g.data && g.data[0] && g.data[0].x);")


class Page:
    """One headless Chrome tab over a file:// page."""

    def __init__(self, page_path, width=1400, height=900, port=None):
        self.uid = tempfile.mkdtemp(prefix='drocat-cdp-')
        # A FREE port, not a random guess in a 100-wide window: a previous
        # test's Chrome that has not finished dying still holds its port, the
        # new Chrome then fails to bind, and the connect loop below would
        # happily drive the OLD page (2026-10-16: the square-cells test read
        # `ticks: 0` from the previous test's heatmap — and a stale page can
        # satisfy a predicate just as easily as fail one).
        if port is None:
            with socket.socket() as _s:
                _s.bind(('127.0.0.1', 0))
                port = _s.getsockname()[1]
        self.port = port
        self.page_url = 'file://' + str(Path(page_path).resolve())
        self.ws = None
        self._id = 0
        self.proc = subprocess.Popen([
            CHROME, f'--user-data-dir={self.uid}', '--headless=new',
            f'--remote-debugging-port={self.port}',
            f'--window-size={width},{height}',
            '--disable-blink-features=AutomationControlled',
            '--allow-file-access-from-files', '--hide-scrollbars',
            'file://' + str(Path(page_path).resolve()),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        while self.ws is None and time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f'http://127.0.0.1:{self.port}/json', timeout=3) as r:
                    targets = [t for t in json.load(r) if t['type'] == 'page']
                # The endpoint must be serving the page WE launched; anything
                # else is a foreign Chrome on the same port, and measuring it
                # is worse than failing.
                target = next((t for t in targets
                               if t.get('url', '').startswith(self.page_url)),
                              None)
                if target is None:
                    raise RuntimeError(
                        'no page for ' + self.page_url + ' (saw '
                        + ', '.join(t.get('url', '?') for t in targets) + ')')
                self.ws = websocket.create_connection(
                    target['webSocketDebuggerUrl'], suppress_origin=True,
                    max_size=None)
            except Exception:
                time.sleep(0.4)
        if self.ws is None:  # pragma: no cover - environment dependent
            self.close()
            pytest.skip(f'Chrome exposed no DevTools endpoint on {self.port}')
        self.ws.settimeout(120)
        self.send('Page.enable')
        self.send('Runtime.enable')
        self.arm_error_watch()

    def send(self, method, params=None):
        self._id += 1
        self.ws.send(json.dumps({'id': self._id, 'method': method,
                                 'params': params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get('id') == self._id:
                if 'error' in msg:
                    raise RuntimeError(msg['error'])
                return msg.get('result', {})

    def js(self, expr):
        """Evaluate JS in the page; return its value (or the exception detail,
        so an assertion sees the throw rather than the harness dying)."""
        out = self.send('Runtime.evaluate', {
            'expression': '(function(){' + expr + '})()',
            'returnByValue': True})
        if out.get('exceptionDetails'):
            return {'__eval_error__':
                    json.dumps(out['exceptionDetails'])[:300]}
        # an expression that returns nothing (a bare call) has no 'value' key
        return out.get('result', {}).get('value')

    def wait(self, predicate_js, timeout=90):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(predicate_js):
                return True
            time.sleep(0.4)
        return False

    def arm_error_watch(self):
        self.js(ARM_ERROR_WATCH)

    def errors(self):
        return self.js("return window.__probeErrors || [];")

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
        shutil.rmtree(self.uid, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
