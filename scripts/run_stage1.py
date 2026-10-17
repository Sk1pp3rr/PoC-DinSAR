#!/usr/bin/env python3
"""
Stage 1: Sentinel-1 DInSAR Pre-processing and Wrapped Interferogram Formation
=============================================================================

Automates the execution of ESA SNAP Graph Processing Tool (GPT) using a multi-step
architecture to prevent GPF tile-cache thrashing:

Step 1A (Coregistration):
  - TOPSAR-Split (isolates sub-swath IW1-3 and target bursts first)
  - Apply-Orbit-File (Precise Orbit Ephemerides POEORB / RESORB)
  - Back-Geocoding (DEM-assisted sub-pixel coregistration)
  - Enhanced Spectral Diversity (ESD, when bursts >= 2)

Step 1B (Interferogram & Deburst):
  - Interferogram Formation (flat-Earth & topographic phase subtraction)
  - TOPSAR-Deburst (merges bursts into a continuous radar swath)

Step 1C (Goldstein Phase Filtering):
  - GoldsteinPhaseFiltering (fast 2D FFT filtering on debursted disk raster)

Step 1D (Optional SnaphuExport):
  - Prepares unwrapping tiles and snaphu.conf for Stage 2.

Author: PoC-DinSAR Team
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage1_snap")

# Standard SNAP install locations on Windows
DEFAULT_SNAP_PATHS = [
    Path(r"C:\Program Files\esa-snap\bin\gpt.exe"),
    Path(r"C:\Program Files\snap\bin\gpt.exe"),
    Path(r"C:\Program Files (x86)\esa-snap\bin\gpt.exe"),
]


def find_gpt_executable(custom_path: Optional[str] = None) -> Path:
    """Find SNAP gpt.exe on system PATH or default Windows installation paths."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified SNAP GPT executable not found: {custom_path}")

    which_gpt = shutil.which("gpt")
    if which_gpt:
        return Path(which_gpt)

    for standard_path in DEFAULT_SNAP_PATHS:
        if standard_path.is_file():
            return standard_path

    raise FileNotFoundError(
        "ESA SNAP 'gpt.exe' could not be found. Please install ESA SNAP or provide --gpt-path."
    )


def extract_s1_timestamp(path: Path) -> str:
    """Extract start timestamp YYYYMMDDTHHMMSS from Sentinel-1 product name."""
    parts = path.name.split("_")
    for part in parts:
        if len(part) == 15 and part[8] == "T" and part[:8].isdigit() and part[9:].isdigit():
            return part
    return path.name


def resolve_product_path(user_input: str, raw_dir: Path) -> Path:
    """Resolve a product path from an exact path, filename, or partial name/timestamp."""
    p = Path(user_input)
    if p.exists():
        return p

    candidate = raw_dir / user_input
    if candidate.exists():
        return candidate

    search_term = p.name.replace(".zip", "").replace(".SAFE", "")
    matches = []
    if raw_dir.is_dir():
        for item in raw_dir.iterdir():
            if search_term in item.name and (item.name.endswith(".zip") or item.name.endswith(".SAFE")):
                matches.append(item)

    if matches:
        safe_matches = [m for m in matches if m.is_dir()]
        return safe_matches[0] if safe_matches else matches[0]

    return p


def find_available_raw_pairs(raw_dir: Path) -> List[Path]:
    """
    Scan raw directory for distinct Sentinel-1 SLC acquisitions.
    Deduplicates between .zip and .SAFE, returning unique scenes sorted chronologically.
    """
    if not raw_dir.is_dir():
        return []

    acquisitions: dict[str, Path] = {}

    for item in raw_dir.iterdir():
        if not item.name.startswith("S1"):
            continue

        if item.is_file() and item.suffix.lower() == ".zip":
            stem = item.name.replace(".zip", "")
            if stem not in acquisitions:
                acquisitions[stem] = item
        elif item.is_dir() and item.name.endswith(".SAFE"):
            stem = item.name.replace(".SAFE", "")
            acquisitions[stem] = item  # .SAFE takes precedence

    sorted_items = sorted(acquisitions.values(), key=extract_s1_timestamp)
    return sorted_items


def run_gpt_graph(
    gpt_exe: Path,
    graph_xml: Path,
    params: dict[str, str],
    step_label: str = "GPT",
    cache_size: str = "6G",
    threads: int = 8,
) -> None:
    """
    Execute a SNAP GPT graph with live unbuffered character streaming so progress
    percentages (..10%..20%..) are logged immediately without waiting for newlines.
    """
    cmd = [
        str(gpt_exe),
        str(graph_xml),
        "-c",
        cache_size,
        "-q",
        str(threads),
    ]

    for key, val in params.items():
        clean_val = str(val).replace("\\", "/")
        cmd.append(f"-P{key}={clean_val}")

    logger.info(f"[{step_label}] Executing graph: {graph_xml.name} (Cache: {cache_size}, Threads: {threads})")
    logger.debug(f"Full command: {' '.join(cmd)}")

    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,  # Unbuffered
    )

    buffer = ""
    last_pct = ""
    pct_pattern = re.compile(r"(\d+%)")

    if process.stdout:
        while True:
            ch = process.stdout.read(1)
            if not ch and process.poll() is not None:
                break
            if not ch:
                continue

            buffer += ch

            # Check for percentage marker (e.g. "..10%..") which SNAP prints without \n
            if ch == "%":
                match = pct_pattern.search(buffer)
                if match:
                    pct_str = match.group(1)
                    if pct_str != last_pct:
                        elapsed_so_far = time.time() - start_time
                        logger.info(f"[{step_label}] Progress: {pct_str} (elapsed: {elapsed_so_far:.0f}s)")
                        last_pct = pct_str
                    buffer = ""

            elif ch == "\n":
                line_str = buffer.strip()
                buffer = ""
                if not line_str or line_str == ".":
                    continue
                if "error" in line_str.lower() or "exception" in line_str.lower():
                    logger.error(f"[{step_label}] {line_str}")
                elif "warning" in line_str.lower():
                    logger.warning(f"[{step_label}] {line_str}")
                else:
                    logger.info(f"[{step_label}] {line_str}")

    process.wait()
    elapsed = time.time() - start_time

    if process.returncode != 0:
        raise RuntimeError(
            f"[{step_label}] SNAP GPT failed with exit code {process.returncode} after {elapsed:.1f}s."
        )

    logger.info(f"[{step_label}] Completed in {elapsed:.1f}s ({elapsed / 60:.2f} min).")


def remove_beam_dimap(dim_path: Path) -> None:
    """Remove a BEAM-DIMAP .dim file and its companion .data directory."""
    if dim_path.is_file():
        dim_path.unlink()
    data_dir = dim_path.with_suffix(".data")
    if data_dir.is_dir():
        shutil.rmtree(data_dir, ignore_errors=True)


def run_stage1_pipeline(
    master_path: Path,
    slave_path: Path,
    output_dir: Path,
    subswath: str = "IW1",
    polarisation: str = "VV",
    first_burst: int = 1,
    last_burst: int = 9,
    aoi_wkt: str = "",
    dem_name: str = "Copernicus 30m Global DEM",
    export_snaphu: bool = False,
    snaphu_export_dir: Optional[Path] = None,
    gpt_path: Optional[str] = None,
    cache_size: str = "6G",
    threads: int = 8,
    cleanup_temp: bool = False,
) -> Path:
    """Run full Stage 1 DInSAR pre-processing pipeline in 3 fast modular sub-steps."""
    gpt_exe = find_gpt_executable(gpt_path)
    logger.info(f"Using SNAP GPT: {gpt_exe}")

    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    graphs_dir = project_root / "graphs"

    m_ts = extract_s1_timestamp(master_path)[:8]
    s_ts = extract_s1_timestamp(slave_path)[:8]
    tag = f"{subswath}_{polarisation}_b{first_burst}_{last_burst}_{m_ts}_{s_ts}"

    # Intermediate & Final BEAM-DIMAP paths
    step1a_dim = output_dir / f"step1a_coreg_{tag}.dim"
    step1b_dim = output_dir / f"step1b_ifg_deburst_{tag}.dim"
    final_dim = output_dir / f"InSAR_Filtered_{tag}.dim"

    total_t0 = time.time()
    logger.info("=====================================================================")
    logger.info("Starting Stage 1: Multi-Step Fast DInSAR Processing Pipeline")
    logger.info(f"  Master SLC (T1): {master_path.name}")
    logger.info(f"  Slave SLC  (T2): {slave_path.name}")
    logger.info(f"  Sub-swath:       {subswath} ({polarisation}), Bursts: {first_burst} to {last_burst}")
    logger.info(f"  Elevation DEM:   {dem_name}")
    logger.info(f"  Final Output:    {final_dim}")
    logger.info("=====================================================================")

    # -------------------------------------------------------------------------
    # STEP 1A: Split -> Orbit -> Back-Geocoding -> ESD
    # -------------------------------------------------------------------------
    if first_burst == last_burst:
        graph_1a = graphs_dir / "stage1a_coregistration_singleburst.xml"
        logger.info(f"[Step 1A] Single-burst mode (burst {first_burst}), ESD bypassed.")
    else:
        graph_1a = graphs_dir / "stage1a_coregistration.xml"
        logger.info(f"[Step 1A] Multi-burst mode (bursts {first_burst}-{last_burst}), ESD enabled.")

    params_1a = {
        "masterFile": str(master_path.resolve()),
        "slaveFile": str(slave_path.resolve()),
        "subswath": subswath,
        "polarisation": polarisation,
        "firstBurstIndex": str(first_burst),
        "lastBurstIndex": str(last_burst),
        "wktAoi": aoi_wkt,
        "demName": dem_name,
        "outputFile": str(step1a_dim.resolve()),
    }
    run_gpt_graph(gpt_exe, graph_1a, params_1a, step_label="Step 1A: Coregistration", cache_size=cache_size, threads=threads)

    # -------------------------------------------------------------------------
    # STEP 1B: Interferogram Formation & TOPSAR-Deburst
    # -------------------------------------------------------------------------
    graph_1b = graphs_dir / "stage1b_ifg_deburst.xml"
    params_1b = {
        "sourceFile": str(step1a_dim.resolve()),
        "polarisation": polarisation,
        "demName": dem_name,
        "outputFile": str(step1b_dim.resolve()),
    }
    run_gpt_graph(gpt_exe, graph_1b, params_1b, step_label="Step 1B: Ifg & Deburst", cache_size=cache_size, threads=threads)

    # -------------------------------------------------------------------------
    # STEP 1C: Goldstein Phase Filtering
    # -------------------------------------------------------------------------
    graph_1c = graphs_dir / "stage1c_goldstein_filter.xml"
    params_1c = {
        "sourceFile": str(step1b_dim.resolve()),
        "outputFile": str(final_dim.resolve()),
    }
    run_gpt_graph(gpt_exe, graph_1c, params_1c, step_label="Step 1C: Goldstein Filter", cache_size=cache_size, threads=threads)

    if not final_dim.is_file():
        raise FileNotFoundError(f"Expected final Stage 1 output was not created: {final_dim}")

    # Optional cleanup of intermediate step 1a & 1b rasters
    if cleanup_temp:
        logger.info("Cleaning up intermediate Step 1A and Step 1B products...")
        remove_beam_dimap(step1a_dim)
        remove_beam_dimap(step1b_dim)

    # -------------------------------------------------------------------------
    # STEP 1D: Optional SnaphuExport
    # -------------------------------------------------------------------------
    if export_snaphu:
        snaphu_target = snaphu_export_dir or (output_dir.parent / "snaphu_export")
        snaphu_target.mkdir(parents=True, exist_ok=True)
        export_graph = graphs_dir / "stage1_export_snaphu.xml"

        snaphu_params = {
            "sourceFile": str(final_dim.resolve()),
            "targetFolder": str(snaphu_target.resolve()),
        }
        run_gpt_graph(
            gpt_exe,
            export_graph,
            snaphu_params,
            step_label="Step 1D: SnaphuExport",
            cache_size=cache_size,
            threads=threads,
        )
        logger.info(f"SNAPHU export files and snaphu.conf created in: {snaphu_target}")

    total_elapsed = time.time() - total_t0
    logger.info("=====================================================================")
    logger.info(f"Stage 1 Pipeline Finished in {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)!")
    logger.info(f"Filtered Interferogram: {final_dim}")
    logger.info("=====================================================================")

    return final_dim


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for Stage 1 execution."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Fast Multi-Step Sentinel-1 DInSAR Processing via SNAP GPT."
    )
    parser.add_argument(
        "--master",
        "-m",
        type=str,
        help="Path or partial name of Master Sentinel-1 SLC (.zip or .SAFE). If omitted, auto-detects from data/raw.",
    )
    parser.add_argument(
        "--slave",
        "-s",
        type=str,
        help="Path or partial name of Slave Sentinel-1 SLC (.zip or .SAFE). If omitted, auto-detects from data/raw.",
    )
    parser.add_argument(
        "--subswath",
        type=str,
        default="IW1",
        choices=["IW1", "IW2", "IW3"],
        help="Target Sentinel-1 sub-swath (default: IW1).",
    )
    parser.add_argument(
        "--polarisation",
        type=str,
        default="VV",
        choices=["VV", "VH", "HH", "HV"],
        help="Polarisation channel to process (default: VV).",
    )
    parser.add_argument(
        "--first-burst",
        type=int,
        default=8,
        help="First burst index (1-based, default: 8 for Gdynia AOI).",
    )
    parser.add_argument(
        "--last-burst",
        type=int,
        default=9,
        help="Last burst index (1-based, default: 9 for Gdynia AOI).",
    )
    parser.add_argument(
        "--aoi-wkt",
        type=str,
        default="",
        help="Optional WKT Polygon to constrain burst selection (EPSG:4326).",
    )
    parser.add_argument(
        "--dem",
        type=str,
        default="Copernicus 30m Global DEM",
        help="Digital Elevation Model for Back-Geocoding & Topo removal (default: Copernicus 30m Global DEM).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/interim/stage1",
        help="Output directory for generated BEAM-DIMAP products (default: data/interim/stage1).",
    )
    parser.add_argument(
        "--export-snaphu",
        action="store_true",
        help="Automatically prepare SNAPHU unwrapping configuration and rasters for Stage 2.",
    )
    parser.add_argument(
        "--cache-size",
        type=str,
        default="6G",
        help="SNAP GPT tile cache size (default: 6G).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of parallel compute threads for SNAP GPT (default: 8).",
    )
    parser.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="Delete intermediate Step 1A and Step 1B .dim/.data files after Step 1C succeeds.",
    )
    parser.add_argument(
        "--gpt-path",
        type=str,
        help="Explicit path to SNAP gpt executable.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    if args.master and args.slave:
        master_path = resolve_product_path(args.master, raw_dir)
        slave_path = resolve_product_path(args.slave, raw_dir)
    elif args.master or args.slave:
        logger.error("Please provide both --master and --slave, or omit both to auto-detect a pair in data/raw.")
        sys.exit(1)
    else:
        logger.info(f"Master and Slave not explicitly provided, searching in {raw_dir}...")
        raw_files = find_available_raw_pairs(raw_dir)
        if len(raw_files) < 2:
            logger.error(
                f"Found {len(raw_files)} distinct acquisition(s) in {raw_dir}. DInSAR requires at least 2 distinct "
                "temporal acquisitions. Please download a pair using scripts/download_cdse.py."
            )
            sys.exit(1)
        master_path = raw_files[0]
        slave_path = raw_files[1]
        logger.info(
            f"Auto-selected distinct chronological pair from {raw_dir}:\n"
            f"  Master (T1): {master_path.name}\n"
            f"  Slave  (T2): {slave_path.name}"
        )

    if not master_path.exists():
        logger.error(f"Master product not found: {master_path}")
        sys.exit(1)
    if not slave_path.exists():
        logger.error(f"Slave product not found: {slave_path}")
        sys.exit(1)

    m_base = master_path.name.replace(".SAFE", "").replace(".zip", "")
    s_base = slave_path.name.replace(".SAFE", "").replace(".zip", "")
    if m_base == s_base:
        logger.error(
            f"Master and Slave cannot be the same acquisition ({m_base})! "
            "InSAR requires two scenes acquired at different dates."
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)

    try:
        run_stage1_pipeline(
            master_path=master_path,
            slave_path=slave_path,
            output_dir=output_dir,
            subswath=args.subswath,
            polarisation=args.polarisation,
            first_burst=args.first_burst,
            last_burst=args.last_burst,
            aoi_wkt=args.aoi_wkt,
            dem_name=args.dem,
            export_snaphu=args.export_snaphu,
            gpt_path=args.gpt_path,
            cache_size=args.cache_size,
            threads=args.threads,
            cleanup_temp=args.cleanup_temp,
        )
    except KeyboardInterrupt:
        logger.info("\nStage 1 interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Stage 1 processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
