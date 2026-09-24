/* Theme resolver + switcher for docs/ui_guides/*.html instruction pages.
 *
 * Mirrors the NiceGUI app's theme (ui/app.py): the app persists its
 * System/Light/Dark choice in a same-origin `drocat_dark` cookie
 * (dark|light|auto, legacy 1|0). This script runs synchronously in <head>,
 * so the resolved theme is on <html data-theme="dark|light"> before first
 * paint — no flash. Without the cookie (file://, cleared) it falls back to
 * the OS prefers-color-scheme, which is what the app's System mode follows.
 *
 * Also injects a fixed top-right toggle styled like the app's header theme
 * button (sun | moon pair). One click flips light/dark: the choice persists
 * to the same drocat_dark cookie and is announced on the 'drocat-theme'
 * BroadcastChannel, so already-open guide pages (and the app, on its next
 * load) follow along. System mode stays in force until a click forces a
 * side; the app's picker restores it.
 */
(function () {
  'use strict';
  var root = document.documentElement;
  var media = window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;

  function savedMode() {
    try {
      var match = document.cookie.match(/(?:^|; )drocat_dark=([^;]*)/);
      var value = match ? decodeURIComponent(match[1]) : '';
      if (value === 'dark' || value === '1') return 'dark';
      if (value === 'light' || value === '0') return 'light';
      return 'auto';
    } catch (err) {
      return 'auto';
    }
  }

  function resolve(mode) {
    if (mode === 'dark' || mode === 'light') return mode;
    return media && media.matches ? 'dark' : 'light';
  }

  function announce(mode) {
    try {
      if (typeof BroadcastChannel !== 'undefined') {
        var channel = new BroadcastChannel('drocat-theme');
        channel.postMessage(mode);
        channel.close();
      }
    } catch (err) { /* same-origin only; ignore environments without it */ }
  }

  var ui = null;

  function syncUI() {
    if (!ui) return;
    var dark = resolve(savedMode()) === 'dark';
    ui.button.setAttribute('aria-pressed', dark ? 'true' : 'false');
    ui.button.setAttribute('title', dark ? 'Switch to light theme' : 'Switch to dark theme');
    ui.button.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }

  function apply() {
    root.setAttribute('data-theme', resolve(savedMode()));
    syncUI();
  }

  function persist(mode) {
    try {
      document.cookie =
        'drocat_dark=' + mode + '; max-age=31536000; path=/; SameSite=Lax';
    } catch (err) { /* file:// etc. — the page still switches */ }
    announce(mode);
    apply();
  }

  /* Inline SVGs: the guides load no icon font, so the toggle is
     self-contained (feather-style glyphs, MIT-licensed shapes). */
  var SVG = {
    light: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    dark: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
  };

  function buildUI() {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'drocat-guide-theme-toggle';
    button.innerHTML =
      '<span class="drocat-gt-icon" aria-hidden="true">' + SVG.light + '</span>' +
      '<span class="drocat-gt-sep" aria-hidden="true"></span>' +
      '<span class="drocat-gt-icon" aria-hidden="true">' + SVG.dark + '</span>';
    button.addEventListener('click', function () {
      persist(resolve(savedMode()) === 'dark' ? 'light' : 'dark');
    });

    var wrap = document.createElement('div');
    wrap.className = 'drocat-guide-theme';
    wrap.appendChild(button);
    document.body.appendChild(wrap);
    ui = { button: button };
    syncUI();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildUI);
  } else {
    buildUI();
  }

  apply();

  // System mode follows the OS preference live.
  if (media && media.addEventListener) {
    media.addEventListener('change', function () {
      if (savedMode() === 'auto') apply();
    });
  }

  // The app (or another guide page) announces a theme change; re-read
  // the cookie it just rewrote.
  try {
    if (typeof BroadcastChannel !== 'undefined') {
      var channel = new BroadcastChannel('drocat-theme');
      channel.onmessage = apply;
    }
  } catch (err) { /* same-origin only; ignore environments without it */ }
})();
