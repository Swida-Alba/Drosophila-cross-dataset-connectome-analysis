# Datasets, authentication, and outputs

## Dataset identifiers

Use the exact identifiers below. Version suffixes are significant.

| Identifier | Source | Notes |
| --- | --- | --- |
| `male-cns:v1.0` | NeuPrint | current male CNS dataset |
| `male-cns:v0.9` | NeuPrint | legacy male CNS; local cache may be useful |
| `hemibrain:v1.2.1` | NeuPrint | adult central brain |
| `optic-lobe:v1.1` | NeuPrint | optic-lobe dataset |
| `manc:v1.2.1` | NeuPrint | male VNC |
| `flywire_FAFB_v783` | FlyWire local files (+ Codex lazy fetch) | requires converted local files; `flywire_codex` token enables Settings downloads and lazy per-neuron skeleton fetch (cached into the `codex_skeleton_cache.zip` overlay); CAVE token is an advanced config-file-only fallback |
| `banc_v888` | Public BANC bucket (auto-prepared) | no token; neuron metadata + connections auto-download on first use; manual Codex files remain an offline fallback |
| `banc_v626` | Public BANC bucket (auto-prepared) | legacy BANC release; same public-bucket preparation |

Validate local FAFB file layout with the repository's integration guides
before running a large query; standalone BANC validity is checked by
`BANC_file_converter.ensure_banc_data` (bucket preparation or local files).
Do not substitute a similarly named dataset silently.

### Exact local input layout

The raw Codex downloads belong under `datasets/<dataset>/downloads/`, not in
the dataset root and not under generated output names.

FAFB downloads can be fetched automatically: save a FlyWire Codex token
(`FLYWIRE_CODEX_TOKEN`, Settings → API Tokens) and use Settings → FAFB Data
Downloads (levels: necessary data / synapses / skeleton bundle). Skeletons
also fetch lazily per neuron from the Codex server during visualization,
so the bundle is optional.

- FAFB v783 requires `classification.csv.gz` and one supported connection file:
  `connections_princeton_no_threshold.csv.gz` (preferred),
  `connections_princeton.csv.gz`, or `connections.csv.gz`.
  `names.csv.gz`, `coordinates.csv.gz`, `neurons.csv.gz`, `cell_stats.csv.gz`,
  and `consolidated_cell_types.csv.gz` enrich the neuron table but are
  optional. Synapse and skeleton files are optional visualization inputs.
- BANC v626/v888 auto-prepares from the public release bucket
  (`banc_public_data.prepare_dataset_tables`: meta feather + per-neuropil
  connection counts; ~134 MB from the bucket, ~186 MB total on disk after conversion; no token) and fills the post-synaptic
  counts from the merged connections. Manual Codex files (`neurons.csv.gz`
  + `connections_princeton.csv.gz` in the matching version's `downloads/`
  folder) remain an offline fallback. Never mix BANC versions.

After conversion, require both `<dataset>_allneurons_neuron_df.parquet` and
`<dataset>_merged_connections.parquet` before calling the dataset prepared.

## Tokens

Preferred sources, in order:

1. `config_local.json` tokens section at the repository root (gitignored override);
2. `config.json` tokens section (committed clean defaults);
3. environment variables handled by the project's token manager;
4. an explicit token argument only when the user has supplied it for this run.

Typical file entries are:

```json
{
  "tokens": {
    "neuprint": "...",
    "flywire_codex": "...",
    "cave": "..."
  },
  "envs": {
    "4.5.0": ""
  }
}
```

`flywire_codex` (Settings → API Tokens) powers the FAFB Dataset Downloads
card and the lazy per-neuron skeleton fetch; `cave` is advanced and
optional (live-CAVE fallback only — there is no Settings input for it).

Never include token contents in logs, patches, notebooks, output reports, or
agent prompts. Use `--require-token` with the install verifier when the user
wants an authentication check.

## Output conventions

Keep outputs outside source directories, preferably under a timestamped
`local_data/agent_runs/<analysis-name>/` folder. Common artifacts include:

- FindPath/FindDirect: CSV/XLSX edge/path tables, summaries, network/heatmap HTML;
- PlotPath: network, Sankey, heatmap HTML plus connection tables;
- 3D skeleton: HTML, parameters, PNG views, individual profiles, PDF/PPTX, GIF/video;
- comparisons: per-dataset tables, threshold summaries, reports, conserved-path HTML;
- NeuronBridge: ranked CSV summaries, expression matrices, images, PDF/PPTX reports.

After a run, list files and inspect schemas. Preserve the run folder until the
user confirms that it can be removed.

## Offline/cache rules

- `use_cache=True` is the default for repeat work.
- `cache_only=True` prevents remote fallback and should be used only when cache
  coverage is known.
- FAFB, standalone BANC, and NeuPrint caches are different resources; one does
  not replace the other.
- A missing token and a missing local file are different failures. Report which
  one occurred.
