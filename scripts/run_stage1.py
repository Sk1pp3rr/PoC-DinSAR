#!/usr/bin/env python3
"""
Stage 1: Sentinel-1 DInSAR Pre-processing and Wrapped Interferogram Formation
=============================================================================

Automates the execution of ESA SNAP Graph Processing Tool (GPT) for:
1. Apply-Orbit-File: Ingests Precise Orbit Ephemerides (POEORB / RESORB).
2. TOPSAR-Split: Isolates target sub-swath (IW1-3) and bursts covering the AOI.
3. Back-Geocoding: Performs sub-pixel coregistration using Copernicus DEM 30m.
4. Enhanced Spectral Diversity (ESD): Corrects burst seam phase discontinuities.
5. Interferogram Formation: Calculates complex phase and subtracts simulated topo phase.
6. TOPSAR-Deburst: Merges bursts into a continuous radar swath.
7. Goldstein Phase Filtering: Suppresses speckle noise to sharpen fringes.
8. (Optional) SnaphuExport: Prepares unwrapping tiles and configuration for Stage 2.

Author: PoC-DinSAR Team
"""

from __future__ import annotations

import argparse
import logging
import os
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

    # Check which/PATH
    which_gpt = shutil.which("gpt") 
    if which_gpt:
        return Path(which_gpt)

    # Check known Windows locations
    for standard_path in DEFAULT_SNAP_PATHS:
        if standard_path.is_file():
            return standard_path

    raise FileNotFoundError(
        "ESA SNAP 'gpt.exe' could not be found. Please install ESA SNAP or provide --gpt-path."
    )


def find_available_raw_pairs(raw_dir: Path) -> List[Path]:
    """Scan raw directory for Sentinel-1 SLC files (.zip or .SAFE)."""
    candidates = []
    if not raw_dir.is_dir():
        return candidates

    for item in raw_dir.iterdir():
        if item.is_file() and item.name.startswith("S1") and item.suffix.lower() == ".zip":
            candidates.append(item)
        elif item.is_dir() and item.name.startswith("S1") and item.name.endswith(".SAFE"):
            candidates.append(item)

    return sorted(candidates)


def run_gpt_graph(
    gpt_exe: Path,
    graph_xml: Path,
    params: dict[str, str],
    vm_memory: str = "10G",
) -> None:
    """
    Execute a SNAP GPT graph with specified key-value parameter substitutions.
    Handles console output streaming and JVM options.
    """
    cmd = [
        str(gpt_exe),
        str(graph_xml),
    ]

    # Convert Windows backslashes to forward slashes for SNAP Java parameter parsing
    for key, val in params.items():
        clean_val = str(val).replace("\\", "/")
        cmd.append(f"-P{key}={clean_val}")

    logger.info(f"Executing SNAP GPT graph: {graph_xml.name}")
    logger.debug(f"Full command: {' '.join(cmd)}")

    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_pct = ""
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            if not line_str:
                continue

            # Capture progress indicator if present (e.g., "..10%..20%..")
            if "%" in line_str and ("." in line_str or line_str.endswith("%")):
                pct_markers = [p.strip() for p in line_str.split(".") if "%" in p]
                if pct_markers:
                    current_pct = pct_markers[-1]
                    if current_pct != last_pct:
                        logger.info(f"Progress: {current_pct}")
                        last_pct = current_pct
                continue

            # Log standard information
            if "error" in line_str.lower() or "exception" in line_str.lower():
                logger.error(f"[GPT] {line_str}")
            elif "warning" in line_str.lower():
                logger.warning(f"[GPT] {line_str}")
            else:
                logger.info(f"[GPT] {line_str}")

    process.wait()
    elapsed = time.time() - start_time

    if process.returncode != 0:
        raise RuntimeError(
            f"SNAP GPT processing failed with exit code {process.returncode} after {elapsed:.1f}s."
        )

    logger.info(f"SNAP GPT execution completed successfully in {elapsed:.1f}s ({elapsed / 60:.2f} min).")


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
) -> Path:
    """Run full Stage 1 DInSAR pre-processing pipeline."""
    gpt_exe = find_gpt_executable(gpt_path)
    logger.info(f"Using SNAP GPT: {gpt_exe}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Master and Slave stem names
    m_stem = master_path.stem.replace(".SAFE", "")
    s_stem = slave_path.stem.replace(".SAFE", "")

    # Output product name
    out_dim_name = f"InSAR_{subswath}_{polarisation}_{first_burst}_{last_burst}_{m_stem[:15]}_{s_stem[:15]}.dim"
    out_dim_path = output_dir / out_dim_name

    # Select appropriate graph XML (ESD requires >= 2 bursts)
    project_root = Path(__file__).resolve().parent.parent
    if first_burst == last_burst:
        graph_xml = project_root / "graphs" / "stage1_dinsar_prep_singleburst.xml"
        logger.info(f"Single-burst mode selected (burst {first_burst}), ESD bypassed.")
    else:
        graph_xml = project_root / "graphs" / "stage1_dinsar_prep.xml"
        logger.info(f"Multi-burst mode selected (bursts {first_burst} to {last_burst}), ESD included.")

    if not graph_xml.is_file():
        raise FileNotFoundError(f"Graph XML template not found: {graph_xml}")

    params = {
        "masterFile": str(master_path.resolve()),
        "slaveFile": str(slave_path.resolve()),
        "subswath": subswath,
        "polarisation": polarisation,
        "firstBurstIndex": str(first_burst),
        "lastBurstIndex": str(last_burst),
        "wktAoi": aoi_wkt,
        "demName": dem_name,
        "outputFile": str(out_dim_path.resolve()),
    }

    logger.info("=== Starting Stage 1: DInSAR Interferogram Generation ===")
    logger.info(f"Master SLC:    {master_path.name}")
    logger.info(f"Slave SLC:     {slave_path.name}")
    logger.info(f"Sub-swath:     {subswath} ({polarisation})")
    logger.info(f"Burst Range:   {first_burst} to {last_burst}")
    logger.info(f"Elevation DEM: {dem_name}")
    logger.info(f"Output Target: {out_dim_path}")

    run_gpt_graph(gpt_exe, graph_xml, params)

    if not out_dim_path.is_file():
        raise FileNotFoundError(f"Expected Stage 1 output file was not created: {out_dim_path}")

    logger.info(f"Stage 1 completed. Filtered wrapped interferogram saved at: {out_dim_path}")

    # Step 8: Optional SnaphuExport
    if export_snaphu:
        snaphu_target = snaphu_export_dir or (output_dir.parent / "snaphu_export")
        snaphu_target.mkdir(parents=True, exist_ok=True)
        export_graph = project_root / "graphs" / "stage1_export_snaphu.xml"

        logger.info("=== Exporting data for Stage 2 (SNAPHU Phase Unwrapping) ===")
        snaphu_params = {
            "sourceFile": str(out_dim_path.resolve()),
            "targetFolder": str(snaphu_target.resolve()),
        }
        run_gpt_graph(gpt_exe, export_graph, snaphu_params)
        logger.info(f"SNAPHU export files and snaphu.conf created in: {snaphu_target}")

    return out_dim_path


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for Stage 1 execution."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Sentinel-1 DInSAR Pre-processing and Wrapped Interferogram Formation via SNAP GPT."
    )
    parser.add_argument(
        "--master",
        "-m",
        type=str,
        help="Path to Master Sentinel-1 SLC (.zip or .SAFE). If omitted, scans data/raw.",
    )
    parser.add_argument(
        "--slave",
        "-s",
        type=str,
        help="Path to Slave Sentinel-1 SLC (.zip or .SAFE). If omitted, scans data/raw.",
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
        default=1,
        help="First burst index (1-based, default: 1).",
    )
    parser.add_argument(
        "--last-burst",
        type=int,
        default=9,
        help="Last burst index (1-based, default: 9).",
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
        help="Output directory for generated BEAM-DIMAP product (default: data/interim/stage1).",
    )
    parser.add_argument(
        "--export-snaphu",
        action="store_true",
        help="Automatically prepare SNAPHU unwrapping configuration and rasters for Stage 2.",
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

    # Resolve Master & Slave paths
    if args.master and args.slave:
        master_path = Path(args.master)
        slave_path = Path(args.slave)
    else:
        logger.info(f"Master and Slave not explicitly provided, searching in {raw_dir}...")
        raw_files = find_available_raw_pairs(raw_dir)
        if len(raw_files) < 2:
            logger.error(
                f"Found {len(raw_files)} product(s) in {raw_dir}. Please provide --master and --slave "
                "or download a pair first using scripts/download_cdse.py."
            )
            sys.exit(1)
        master_path = raw_files[0]
        slave_path = raw_files[1]
        logger.info(f"Auto-selected pair from {raw_dir}:\n  Master: {master_path.name}\n  Slave:  {slave_path.name}")

    if not master_path.exists():
        logger.error(f"Master product not found: {master_path}")
        sys.exit(1)
    if not slave_path.exists():
        logger.error(f"Slave product not found: {slave_path}")
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
        )
    except Exception as e:
        logger.error(f"Stage 1 processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
