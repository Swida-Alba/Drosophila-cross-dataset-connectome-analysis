# FlyLight Downloader (flylight)

Reproduce the **FlyLight Downloader** UI tab as a direct backend call. Uses
`FlyLightDownloader` to download expression images/metadata for driver lines,
optionally generating PDF/PPTX summaries.

## Backend contract

- **tool_key:** `flylight_download`
- **import:** `from flylight_downloader import FlyLightDownloader`
- **class:** `FlyLightDownloader` (var `downloader`)
- **method:** `downloader.download(**method_params)`

## Parameters the UI builds

```python
from flylight_downloader import FlyLightDownloader

downloader = FlyLightDownloader(
    output_dir="/absolute/output/flylight",
    formats=["png"],                    # ["png", "jpg", "tif", ...]
    image_types=["mip"],
    region=None,                        # filter by brain region
    collection_category=None,           # "GAL4/LEXA" | "SplitGAL4" | "MCFO" | "RawImages" | "All"
    max_workers=4,
    simple_mode=False,
    use_boto3=True,
    include_vt_lines=True,
    verbose="pbar",                     # "pbar" | True | False
)

result = downloader.download(
    line_name=["SS00001"],              # or a list of line names
    output_dir="/absolute/output/flylight",
    max_files=None,                     # bound downloads (small first)
    flat_structure=False,
    add_timestamp=True,
    generate_summary=None,              # "pdf" | "pptx" | ["pdf","pptx"] | None
    summary_images_per_page=(3, 2),
)
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_FlyLight_<date>.py
```

## Outputs

- Image/metadata files under the output folder; optional PDF/PPTX summary reports.

## Notes

- Start with `dry_run` and a small `max_files` (the backend supports a dry-run flag;
  a tiny `max_files` limits how many images are downloaded).
- `simple_mode=True` skips broad category fallbacks for a faster, narrower download.
- `generate_summary` accepts a string or a list (`["pdf","pptx"]`).
