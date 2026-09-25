#!/usr/bin/env python3
"""
FlyLight_fetcher.py - Download FlyLight imagery data by driver line name

This script downloads images and data from multiple FlyLight sources:
    1. S3 bucket (janelia-flylight-imagery): Gen1 MCFO, Split-GAL4 lines
    2. HTTP CDN (flimg.janelia.org): VT GAL4 lines

The script automatically detects the line type and uses the appropriate source.

Usage:
    Edit the parameters in the script and run directly:
    python FlyLight_fetcher.py

Key Features:
    - Download by driver line name (e.g., 'R10A06', 'VT037867', 'SS00731')
    - Automatic source detection (S3 for R-lines, HTTP for VT lines)
    - Filter by file format: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', or 'all'
    - Filter by image type: 'mip', 'cdm', 'aligned', 'metadata', etc.
    - List files without downloading (dry_run mode)
    - Search for lines matching a pattern
    - Parallel downloads for speed

File Formats:
    - png: PNG images (MIP, CDM, aligned stack previews)
    - jpg: JPEG images (VT line projections)
    - h5j: H5J 3D stacks (aligned and unaligned)
    - lsm: Raw confocal data (compressed .lsm.bz2)
    - mp4: Translation videos
    - json: Metadata files

"""
# DROCAT edit-to-run TEMPLATE, not a CLI. It mirrors the UI's
# "flylight_download" tool (registry
# ui/runner.py TOOL_REGISTRY; payload built in ui/tabs/flylight.py), and its
# values below are pinned to ui/config.py DEFAULTS so an unedited
# run reproduces what the tab sends. Copy to
# archive/scripts_local/ before a real scientific run
# (skills/drocat-usage/SKILL.md:66); the relative paths assume a
# working directory of scripts/.

import sys
from pathlib import Path

# Add repo src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from flylight_downloader import FlyLightDownloader

if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURATION - Edit these parameters
    # ==========================================================================
    
    # Driver line name(s) to download
    # Examples: 
    #   'R10A06'   - Gen1 line (from S3 bucket)
    #   'VT037867' - VT line (from HTTP CDN - automatically detected)
    #   'SS00731'  - Split-GAL4 line (from S3 bucket)
    # Multiple lines (comma-separated or list):
    #   'SS01015,VT037867' - Download from multiple lines
    #   ['SS01015', 'VT037867'] - Same as above using list
    line_name = ['SS01015', 'VT037867']
    
    # File formats to download
    # Options: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', 'all'
    # Note: VT lines primarily have 'jpg' and 'mp4' files
    # Can be a single format or list: ['png', 'jpg', 'json']
    formats = ['jpg', 'png']
    
    # Collection category (case-insensitive, alternative to collections)
    # Options: 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'
    # Can be a single category or list: ['GAL4/LEXA', 'SplitGAL4']
    collection_category = ['GAL4/LexA', 'SplitGAL4']  # None = search all collections
    
    # Region filter - filter by anatomical region
    # Options: 'Brain', 'VNC', or 'All' (default)
    region = 'Brain'
    
    # Output directory for downloaded files
    output_dir = '../local_data/flylight'
    
    # Maximum number of files to download per line (None = no limit)
    # When downloading multiple lines, this limit applies to each line separately
    max_files = 6
    
    # Number of parallel download threads
    max_workers = 4
    
    # Verbose output
    verbose = True
    
    # Generate image summary (PDF/PPTX) after download
    # Options:
    #   - 'pdf' or ['pdf']: Generate PDF only
    #   - 'pptx' or ['pptx']: Generate PPTX only
    #   - ['pdf', 'pptx']: Generate both formats
    #   - True: Generate both formats
    #   - False or None: Disable generation
    generate_summary = ['pdf', 'pptx']
    
    # Images per page/slide for summary (columns, rows)
    summary_images_per_page = (3, 2)
    
    # Add timestamp to output folder name (creates {line_name}_{timestamp}/)
    add_timestamp = True
    
    # Background color for summary (PDF/PPTX)
    # Options: 'black' (default), 'white', '#000000', '#FFFFFF', etc.
    background_color = 'black'
    
    # Verbose output: True for full output, 'pbar' for progress bar only
    verbose = True
    
    # ==========================================================================
    # EXECUTION - No need to edit below this line
    # ==========================================================================
    
    # Initialize downloader
    downloader = FlyLightDownloader(
        output_dir=output_dir,
        collection_category=collection_category,
        region=region,
        max_workers=max_workers,
        verbose=verbose,
        simple_mode=True, # Enable simple mode for reduced download volume
    )
    
    # Download files with summary generation
    downloaded_files = downloader.download(
        line_name=line_name,
        max_files=max_files,
        flat_structure=True,
        generate_summary=generate_summary,
        summary_images_per_page=summary_images_per_page,
        add_timestamp=add_timestamp,
        background_color=background_color,
    )
    
    # Summary
    if downloaded_files:
        print(f"\n{'='*70}")
        print(f"📁 Downloaded {len(downloaded_files)} files")
        print("\n   Files:")
        for f in downloaded_files[:10]:
            print(f"   - {f.name}")
        if len(downloaded_files) > 10:
            print(f"   ... and {len(downloaded_files) - 10} more")
    else:
        print(f"\n⚠️  No files downloaded for '{line_name}'")
        print("   Try different filters or check the line name.")
    
    print("\n✅ Done!")
    
    # ==========================================================================
    # ADDITIONAL USAGE EXAMPLES (uncomment to use)
    # ==========================================================================
    
    # # Example 1: Search for lines matching a pattern
    # matching_lines = downloader.search_lines('R10A.*')
    # print(f"Found {len(matching_lines)} lines matching 'R10A.*'")
    # for line in matching_lines[:20]:
    #     print(f"  - {line}")
    
    # # Example 2: Get metadata without downloading images
    # metadata = downloader.get_metadata('R10A06')
    # print(f"Found {len(metadata)} metadata records")
    # if metadata:
    #     print(f"First record: {metadata[0]}")
    
    # # Example 3: List all files without downloading
    # files = downloader.list_files('R10A06')
    # print(f"Found {len(files)} total files")
    # for f in files[:20]:
    #     print(f"  - {f.filename} ({f.size_mb:.1f} MB)")
    
    # # Example 4: Download aligned stacks (H5J format)
    # downloader.formats = ['h5j']
    # downloader.image_types = ['aligned']
    # downloaded = downloader.download('R10A06', max_files=2)
    
    # # Example 5: Download all CDM masks for NeuronBridge
    # downloader.formats = ['png']
    # downloader.image_types = ['cdm']
    # downloaded = downloader.download('SS00731')
