#!/usr/bin/env python3
"""One-off re-run of the nb_colabel tool with a larger timeout budget.

The production suite caps nb_colabel at 900s; the run on 2026-09-01 reached
93% (42/45 images) downloading NeuronBridge CDN images when the timeout
fired. This proves the tool completes given a realistic budget.
"""
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.runner import ScriptRunner  # noqa: E402

OUT = Path("/tmp/drocat_e2e/nb_colabel_rerun")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runner = ScriptRunner()
    logs = []

    started = time.time()
    try:
        res = await asyncio.wait_for(
            runner.run(
                "nb_colabel",
                {"verbose": True},
                "colabel",
                method_params={
                    "lines": ["VT037867", "R10A06"],
                    "output_dir": str(OUT),
                    "similarity_methods": ["jaccard", "weighted_jaccard"],
                    "generate_report": True,
                    "visualize": True,
                    "visualize_top_n": 0,
                },
                log_callback=lambda line, stream="stdout": logs.append(line),
                output_dir=str(OUT),
            ),
            timeout=1800,
        )
        status = "SUCCESS" if res["returncode"] == 0 and not res["cancelled"] else "FAILED"
        files = list(OUT.rglob("*"))
        n_files = sum(1 for f in files if f.is_file())
        print(f"[{status}] nb_colabel {time.time() - started:.1f}s "
              f"returncode={res['returncode']} files={n_files}")
    except asyncio.TimeoutError:
        runner.cancel()
        print(f"[TIMEOUT] nb_colabel still exceeded 1800s")


if __name__ == "__main__":
    asyncio.run(main())
