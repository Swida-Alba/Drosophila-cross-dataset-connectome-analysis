# Windows round-6 test harness (as executed)

The exact harness the Windows round-6 tester ran, committed verbatim so
round 7 does not reverse-engineer it (round-6 report §R7 item 2). The
`t_*.py` probes are self-contained: they build synthetic caches/tables
under a scratch root and drive the public surfaces. Machine-specific
roots (`E:\DROCAT_test_260925\...`) appear inside — adjust the constants
at the top of each script to the round-7 host.

- `t_cache_gate.py` — §B matrix (refusal/certify/growth) on a synthetic
  NeuPrint-shaped cache; includes the round-5 `t_normal_run` /
  `t_cacheonly` / `t_truncate_cache` scenarios the old plan referenced.
- `t_windows_file_layer.py` — §C re-encode/compaction/temp-hygiene rows.
- `t_encoding_isolation.py` — F-D1 isolation (bare import vs
  import-coana-first vs PYTHONIOENCODING child).
- `t_banc_prepare.py` — §C2/O13 BANC public-bucket preparation (exercises
  the F-N1 resume path on a lossy link).
- `t_e_listing.py` — §E dataset-catalog probes.
- `t_fafb_j.py`, `t_j_reassert.py`, `t_j5_j6.py` — §J real-data rows.
- `t_online_small.py` — the O2/O3 online probes (quantified abort cost).
- `probe_findstr_predicate.bat` — the F-P1 isolation table.
- `a6d_foreign_owner_leg1.cmd` — A6d's non-DROCAT owner leg.

Round-6 report: `docs/audits/` (local) / the zipped
`DROCAT_retest_report_round6_2026-09-25.md`.
