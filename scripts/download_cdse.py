#!/usr/bin/env python3
"""
Copernicus Data Space Ecosystem (CDSE) Downloader for Sentinel-1 IW SLC Data
=============================================================================

Automated search and download client for Sentinel-1 Interferometric Wide (IW)
Single Look Complex (SLC) products from the Copernicus Data Space Ecosystem API.
Designed for Stage 1 of the Sentinel-1 DInSAR processing pipeline.

Features:
- OAuth2 Keycloak authentication with automatic token refresh.
- Spatial (AOI / Bounding Box) and temporal filtering.
- Sentinel-1 IW SLC filtering (Track / Relative Orbit, Orbit Direction).
- DInSAR pair identification (Master / Slave matching by relative orbit & baseline).
- Resumable multi-gigabyte downloads with chunk streaming and tqdm progress bar.
- Correct handling of CDSE Zipper redirects (preserving Authorization header).
- Optional automatic archive extraction to `.SAFE` directory for SNAP GPT.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_cdse")

# CDSE API Endpoints
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CLIENT_ID = "cdse-public"

# Presets for common study areas (e.g. Redłowo Cliff / Gdynia, Poland)
PRESET_AOIS: Dict[str, Tuple[float, float, float, float]] = {
    "gdynia": (18.45, 54.45, 18.65, 54.55),  # min_lon, min_lat, max_lon, max_lat
}


@dataclass
class S1Product:
    """Represents a Sentinel-1 product metadata entity from CDSE OData."""
    id: str
    name: str
    content_date_start: str
    content_date_end: str
    size_bytes: int
    relative_orbit: Optional[int] = None
    orbit_direction: Optional[str] = None
    polarisation: Optional[str] = None
    sensor_mode: Optional[str] = "IW"
    product_type: Optional[str] = "SLC"
    footprint: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def start_datetime(self) -> datetime.datetime:
        cleaned = self.content_date_start.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(cleaned)


class CDSEDownloader:
    """Client for Copernicus Data Space Ecosystem (CDSE) authentication, catalogue search, and download."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        output_dir: str | Path = "data/raw",
    ):
        load_dotenv()
        self.username = username or os.getenv("CDSE_USERNAME") or os.getenv("CDSE_EMAIL")
        self.password = password or os.getenv("CDSE_PASSWORD")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None

    def authenticate(self, force_refresh: bool = False) -> str:
        """Authenticate with CDSE Keycloak service and acquire / refresh OAuth2 access token."""
        now = datetime.datetime.now().timestamp()
        if (
            not force_refresh
            and self._access_token
            and self._token_expires_at
            and (self._token_expires_at - now) > 60
        ):
            return self._access_token

        # Try refresh token if available
        if self._refresh_token and not force_refresh:
            try:
                payload = {
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": self._refresh_token,
                }
                res = self._session.post(TOKEN_URL, data=payload, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    self._access_token = data["access_token"]
                    self._refresh_token = data.get("refresh_token", self._refresh_token)
                    self._token_expires_at = now + data.get("expires_in", 600)
                    logger.debug("Refreshed CDSE access token successfully.")
                    return self._access_token
            except Exception as e:
                logger.debug(f"Refresh token attempt failed ({e}), falling back to password grant.")

        # Full password grant
        if not self.username or not self.password:
            raise ValueError(
                "CDSE credentials missing. Please set CDSE_USERNAME and CDSE_PASSWORD in .env, "
                "or export them as environment variables, or pass --username and --password."
            )

        payload = {
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "username": self.username,
            "password": self.password,
        }
        res = self._session.post(TOKEN_URL, data=payload, timeout=20)
        if res.status_code != 200:
            error_msg = res.text
            try:
                error_msg = res.json().get("error_description", error_msg)
            except Exception:
                pass
            raise PermissionError(f"CDSE authentication failed (HTTP {res.status_code}): {error_msg}")

        data = res.json()
        self._access_token = data["access_token"]
        self._refresh_token = data.get("refresh_token")
        self._token_expires_at = now + data.get("expires_in", 600)
        logger.info(f"Authenticated successfully as {self.username}.")
        return self._access_token

    def search(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        aoi_wkt: Optional[str] = None,
        track: Optional[int] = None,
        orbit_direction: Optional[str] = None,
        product_name: Optional[str] = None,
        product_id: Optional[str] = None,
        top: int = 20,
    ) -> List[S1Product]:
        """
        Query CDSE OData catalogue for Sentinel-1 IW SLC products.

        :param start_date: ISO date string, e.g., '2024-01-01' or '2024-01-01T00:00:00.000Z'
        :param end_date: ISO date string, e.g., '2024-02-01' or '2024-02-01T23:59:59.000Z'
        :param bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
        :param aoi_wkt: Polygon WKT, e.g., "POLYGON((lon lat, ...))"
        :param track: Relative orbit number (1-175)
        :param orbit_direction: 'ASCENDING' or 'DESCENDING'
        :param product_name: Exact name or substring match
        :param product_id: CDSE Product UUID
        :param top: Maximum number of results to retrieve (default: 20)
        :return: List of S1Product objects
        """
        filters: List[str] = ["Collection/Name eq 'SENTINEL-1'"]

        if product_id:
            filters.append(f"Id eq {product_id}")
        elif product_name:
            if product_name.endswith(".SAFE") or product_name.endswith(".zip"):
                clean_name = product_name.replace(".zip", "")
                if not clean_name.endswith(".SAFE"):
                    clean_name += ".SAFE"
                filters.append(f"Name eq '{clean_name}'")
            else:
                filters.append(f"contains(Name, '{product_name}')")
        else:
            filters.append("contains(Name, 'IW_SLC')")

        if start_date:
            dt_str = start_date if "T" in start_date else f"{start_date}T00:00:00.000Z"
            filters.append(f"ContentDate/Start ge {dt_str}")

        if end_date:
            dt_str = end_date if "T" in end_date else f"{end_date}T23:59:59.999Z"
            filters.append(f"ContentDate/Start le {dt_str}")

        # Spatial geometry
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            wkt = (
                f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
                f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
            )
            filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')")
        elif aoi_wkt:
            filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')")

        # Track (Relative Orbit Number)
        if track is not None:
            filters.append(
                f"Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq 'relativeOrbitNumber' "
                f"and att/OData.CSC.IntegerAttribute/Value eq {track})"
            )

        # Orbit direction
        if orbit_direction:
            filters.append(
                f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' "
                f"and att/OData.CSC.StringAttribute/Value eq '{orbit_direction.upper()}')"
            )

        query_filter = " and ".join(filters)
        params = {
            "$filter": query_filter,
            "$top": top,
            "$orderby": "ContentDate/Start asc",
            "$expand": "Attributes",
        }

        logger.info(f"Querying CDSE Catalogue with {len(filters)} filters...")
        res = self._session.get(ODATA_BASE_URL, params=params, timeout=30)
        if res.status_code != 200:
            raise RuntimeError(f"CDSE catalogue query failed (HTTP {res.status_code}): {res.text}")

        data = res.json()
        raw_items = data.get("value", [])
        products: List[S1Product] = []

        for item in raw_items:
            attrs = {a.get("Name"): a.get("Value") for a in item.get("Attributes", [])}
            prod = S1Product(
                id=item["Id"],
                name=item["Name"],
                content_date_start=item.get("ContentDate", {}).get("Start", ""),
                content_date_end=item.get("ContentDate", {}).get("End", ""),
                size_bytes=item.get("ContentLength", 0),
                relative_orbit=attrs.get("relativeOrbitNumber"),
                orbit_direction=attrs.get("orbitDirection"),
                polarisation=attrs.get("polarisationChannels"),
                sensor_mode=attrs.get("operationalMode", "IW"),
                product_type=attrs.get("productType", "SLC"),
                footprint=item.get("GeoFootprint"),
                attributes=attrs,
            )
            products.append(prod)

        logger.info(f"Found {len(products)} matching product(s).")
        return products

    def find_pairs(
        self,
        products: List[S1Product],
        min_baseline_days: int = 6,
        max_baseline_days: int = 48,
    ) -> List[Dict[str, Any]]:
        """
        Group products by Relative Orbit (track) and find compatible DInSAR Master-Slave pairs.

        :param products: List of S1Product objects
        :param min_baseline_days: Minimum temporal baseline (default: 6 days)
        :param max_baseline_days: Maximum temporal baseline (default: 48 days)
        :return: List of candidate pairs with temporal baseline info
        """
        grouped: Dict[Tuple[Optional[int], Optional[str]], List[S1Product]] = {}
        for p in products:
            key = (p.relative_orbit, p.orbit_direction)
            grouped.setdefault(key, []).append(p)

        pairs: List[Dict[str, Any]] = []
        for (rel_orbit, direction), group in grouped.items():
            # Sort chronologically
            sorted_group = sorted(group, key=lambda x: x.start_datetime)
            n = len(sorted_group)
            for i in range(n):
                for j in range(i + 1, n):
                    master = sorted_group[i]
                    slave = sorted_group[j]
                    delta = (slave.start_datetime - master.start_datetime).total_seconds() / 86400.0
                    days = round(delta)

                    if min_baseline_days <= days <= max_baseline_days:
                        pairs.append({
                            "track": rel_orbit,
                            "direction": direction,
                            "temporal_baseline_days": days,
                            "master": master,
                            "slave": slave,
                        })

        return pairs

    def download_product(
        self,
        product: S1Product | str,
        product_name: Optional[str] = None,
        extract: bool = False,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> Path:
        """
        Download a Sentinel-1 product by ID or S1Product object with resumable transfer.

        :param product: S1Product object or CDSE Product UUID string
        :param product_name: File name (optional if passing S1Product)
        :param extract: If True, automatically unzip into .SAFE directory
        :param chunk_size: Stream chunk size in bytes (default: 8 MB)
        :return: Path to the downloaded .zip or extracted .SAFE directory
        """
        token = self.authenticate()

        if isinstance(product, S1Product):
            prod_id = product.id
            prod_name = product.name
            expected_size = product.size_bytes
        else:
            prod_id = product
            prod_name = product_name or f"{prod_id}.SAFE"
            expected_size = 0

        # Standardize target archive filename
        archive_name = prod_name if prod_name.endswith(".zip") else f"{prod_name.rstrip('.SAFE')}.zip"
        dest_zip = self.output_dir / archive_name
        dest_safe = self.output_dir / (archive_name.replace(".zip", ".SAFE"))
        part_file = self.output_dir / f"{archive_name}.part"

        # Check if already extracted or downloaded
        if extract and dest_safe.is_dir():
            logger.info(f"Product already extracted at: {dest_safe}")
            return dest_safe

        if dest_zip.is_file() and not part_file.exists():
            current_size = dest_zip.stat().st_size
            if expected_size == 0 or current_size >= expected_size:
                logger.info(f"Product already downloaded: {dest_zip} ({current_size / (1024**3):.2f} GB)")
                if extract:
                    return self._extract_archive(dest_zip, dest_safe)
                return dest_zip

        # Download URL
        download_url = f"{ODATA_BASE_URL}({prod_id})/$value"

        # Check existing partial file for resume
        existing_bytes = 0
        if part_file.exists():
            existing_bytes = part_file.stat().st_size
            logger.info(f"Resuming download from byte {existing_bytes} ({existing_bytes / (1024**3):.2f} GB)...")

        headers = {
            "Authorization": f"Bearer {token}",
        }
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        # Execute GET request manually handling redirects to preserve Authorization header
        logger.info(f"Connecting to download stream for: {prod_name}")
        req_url = download_url
        response = None

        for redirect_hop in range(5):
            res = self._session.get(req_url, headers=headers, stream=True, allow_redirects=False, timeout=60)
            if res.status_code in (301, 302, 303, 307, 308):
                redirect_target = res.headers.get("Location")
                if not redirect_target:
                    raise RuntimeError(f"Redirect received without Location header (HTTP {res.status_code})")
                req_url = redirect_target
                logger.debug(f"Redirect hop {redirect_hop + 1} to: {req_url}")
                continue
            elif res.status_code == 401:
                logger.warning("Token expired during download initialization, re-authenticating...")
                token = self.authenticate(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                res = self._session.get(req_url, headers=headers, stream=True, allow_redirects=False, timeout=60)

            response = res
            break

        if response is None or response.status_code not in (200, 206):
            status = response.status_code if response else "Unknown"
            err_body = response.text[:500] if response else ""
            raise RuntimeError(f"Failed to initiate download for {prod_name} (HTTP {status}): {err_body}")

        # Determine total size
        content_length = response.headers.get("Content-Length")
        total_size = int(content_length) + existing_bytes if content_length else expected_size

        mode = "ab" if existing_bytes > 0 else "wb"
        logger.info(f"Downloading {prod_name} ({total_size / (1024**3):.2f} GB)...")

        with open(part_file, mode) as f, tqdm(
            total=total_size,
            initial=existing_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=archive_name[:32],
            miniters=1,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Atomic rename to final zip file
        if part_file.exists():
            part_file.replace(dest_zip)
            logger.info(f"Saved completed archive to: {dest_zip}")

        if extract:
            return self._extract_archive(dest_zip, dest_safe)

        return dest_zip

    def _extract_archive(self, zip_path: Path, target_safe_path: Path) -> Path:
        """Extract a Sentinel-1 .zip archive into .SAFE directory."""
        logger.info(f"Extracting {zip_path.name} to {self.output_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.output_dir)
        logger.info(f"Extraction complete: {target_safe_path}")
        return target_safe_path

    def download_pair(
        self,
        master: S1Product | str,
        slave: S1Product | str,
        extract: bool = False,
    ) -> Dict[str, Path]:
        """
        Download a Master and Slave Sentinel-1 SLC pair for DInSAR processing.

        :param master: Master product (S1Product or ID)
        :param slave: Slave product (S1Product or ID)
        :param extract: If True, extract archives to .SAFE folders
        :return: Dict with 'master_path' and 'slave_path'
        """
        logger.info("=== Downloading Master Scene ===")
        master_path = self.download_product(master, extract=extract)
        logger.info("=== Downloading Slave Scene ===")
        slave_path = self.download_product(slave, extract=extract)

        logger.info("Pair download completed successfully.")
        return {
            "master_path": master_path,
            "slave_path": slave_path,
        }


def format_products_table(products: List[S1Product]) -> str:
    """Format product list into a readable CLI summary table."""
    lines = [
        "-" * 110,
        f"{'Index':<6} {'Acquisition Start':<20} {'Track':<6} {'Dir':<5} {'Size (GB)':<10} {'Product Name':<50}",
        "-" * 110,
    ]
    for idx, p in enumerate(products, 1):
        dt_str = p.content_date_start[:19].replace("T", " ")
        track_str = str(p.relative_orbit) if p.relative_orbit else "N/A"
        dir_str = (p.orbit_direction or "N/A")[:4]
        lines.append(
            f"{idx:<6} {dt_str:<20} {track_str:<6} {dir_str:<5} {p.size_gb:<10.2f} {p.name:<50}"
        )
    lines.append("-" * 110)
    return "\n".join(lines)


def format_pairs_table(pairs: List[Dict[str, Any]]) -> str:
    """Format candidate DInSAR pairs into a readable summary table."""
    lines = [
        "=" * 115,
        f"{'Pair #':<7} {'Track':<6} {'Dir':<5} {'Delta (days)':<13} {'Master Acquisition':<20} {'Slave Acquisition':<20} {'Master ID':<36}",
        "=" * 115,
    ]
    for idx, pair in enumerate(pairs, 1):
        m: S1Product = pair["master"]
        s: S1Product = pair["slave"]
        m_dt = m.content_date_start[:19].replace("T", " ")
        s_dt = s.content_date_start[:19].replace("T", " ")
        lines.append(
            f"{idx:<7} {str(pair['track']):<6} {str(pair['direction'])[:4]:<5} {pair['temporal_baseline_days']:<13} "
            f"{m_dt:<20} {s_dt:<20} {m.id:<36}"
        )
    lines.append("=" * 115)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Sentinel-1 IW SLC Downloader from Copernicus Data Space Ecosystem (CDSE) for DInSAR."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Common options helper
    def add_common_search_args(p: argparse.ArgumentParser):
        p.add_argument("--start-date", "-s", type=str, help="Start date (YYYY-MM-DD)")
        p.add_argument("--end-date", "-e", type=str, help="End date (YYYY-MM-DD)")
        p.add_argument(
            "--bbox",
            nargs=4,
            type=float,
            metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
            help="Bounding box coordinates in WGS84 EPSG:4326",
        )
        p.add_argument("--aoi-wkt", type=str, help="AOI polygon WKT (EPSG:4326)")
        p.add_argument("--preset", choices=list(PRESET_AOIS.keys()), help="Predefined study area (e.g. gdynia)")
        p.add_argument("--track", type=int, help="Sentinel-1 Relative Orbit Number (1-175)")
        p.add_argument(
            "--orbit-direction",
            choices=["ASCENDING", "DESCENDING"],
            help="Orbit pass direction",
        )
        p.add_argument("--top", type=int, default=20, help="Maximum number of products to fetch (default: 20)")

    # 1. search subcommand
    p_search = subparsers.add_parser("search", help="Search CDSE catalogue for Sentinel-1 IW SLC products.")
    add_common_search_args(p_search)
    p_search.add_argument("--name", type=str, help="Product name or substring filter")

    # 2. find-pairs subcommand
    p_pairs = subparsers.add_parser(
        "find-pairs",
        help="Search and identify compatible DInSAR Master-Slave pairs (matching orbit & short baseline).",
    )
    add_common_search_args(p_pairs)
    p_pairs.add_argument("--min-baseline", type=int, default=6, help="Min temporal baseline in days (default: 6)")
    p_pairs.add_argument("--max-baseline", type=int, default=48, help="Max temporal baseline in days (default: 48)")

    # 3. download subcommand
    p_download = subparsers.add_parser("download", help="Download specific product(s) by name, ID, or search.")
    add_common_search_args(p_download)
    p_download.add_argument("--id", type=str, help="CDSE Product UUID to download")
    p_download.add_argument("--name", type=str, help="Exact product name (or .SAFE/.zip) to download")
    p_download.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/raw",
        help="Directory to save downloads (default: data/raw)",
    )
    p_download.add_argument(
        "--extract",
        action="store_true",
        help="Automatically unpack downloaded .zip archive to .SAFE directory",
    )
    p_download.add_argument("--username", "-u", type=str, help="CDSE username / email")
    p_download.add_argument("--password", "-p", type=str, help="CDSE password")

    # 4. download-pair subcommand
    p_dl_pair = subparsers.add_parser(
        "download-pair",
        help="Search and download the first matching DInSAR Master-Slave pair, or specify IDs.",
    )
    add_common_search_args(p_dl_pair)
    p_dl_pair.add_argument("--master-id", type=str, help="Master scene CDSE UUID")
    p_dl_pair.add_argument("--slave-id", type=str, help="Slave scene CDSE UUID")
    p_dl_pair.add_argument("--min-baseline", type=int, default=6, help="Min temporal baseline in days (default: 6)")
    p_dl_pair.add_argument("--max-baseline", type=int, default=36, help="Max temporal baseline in days (default: 36)")
    p_dl_pair.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/raw",
        help="Directory to save downloads (default: data/raw)",
    )
    p_dl_pair.add_argument(
        "--extract",
        action="store_true",
        help="Automatically unpack downloaded .zip archive to .SAFE directory",
    )
    p_dl_pair.add_argument("--username", "-u", type=str, help="CDSE username / email")
    p_dl_pair.add_argument("--password", "-p", type=str, help="CDSE password")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Resolve bounding box
    bbox = None
    if hasattr(args, "preset") and args.preset:
        bbox = PRESET_AOIS.get(args.preset)
    elif hasattr(args, "bbox") and args.bbox:
        bbox = tuple(args.bbox)

    username = getattr(args, "username", None)
    password = getattr(args, "password", None)
    output_dir = getattr(args, "output_dir", "data/raw")
    client = CDSEDownloader(username=username, password=password, output_dir=output_dir)

    if args.command == "search":
        products = client.search(
            start_date=args.start_date,
            end_date=args.end_date,
            bbox=bbox,
            aoi_wkt=args.aoi_wkt,
            track=args.track,
            orbit_direction=args.orbit_direction,
            product_name=args.name,
            top=args.top,
        )
        if not products:
            logger.info("No products found matching the criteria.")
            return
        print("\n" + format_products_table(products))

    elif args.command == "find-pairs":
        products = client.search(
            start_date=args.start_date,
            end_date=args.end_date,
            bbox=bbox,
            aoi_wkt=args.aoi_wkt,
            track=args.track,
            orbit_direction=args.orbit_direction,
            top=args.top,
        )
        pairs = client.find_pairs(
            products,
            min_baseline_days=args.min_baseline,
            max_baseline_days=args.max_baseline,
        )
        if not pairs:
            logger.info("No matching pairs found within the specified baseline constraints.")
            return
        print(f"\nFound {len(pairs)} candidate DInSAR pair(s):")
        print(format_pairs_table(pairs))

    elif args.command == "download":
        if args.id:
            logger.info(f"Starting download for product ID: {args.id}")
            path = client.download_product(args.id, extract=args.extract)
            logger.info(f"Download complete: {path}")
        elif args.name:
            products = client.search(product_name=args.name, top=1)
            if not products:
                logger.error(f"Could not find product matching: {args.name}")
                sys.exit(1)
            path = client.download_product(products[0], extract=args.extract)
            logger.info(f"Download complete: {path}")
        else:
            products = client.search(
                start_date=args.start_date,
                end_date=args.end_date,
                bbox=bbox,
                aoi_wkt=args.aoi_wkt,
                track=args.track,
                orbit_direction=args.orbit_direction,
                top=args.top,
            )
            if not products:
                logger.error("No products found to download.")
                sys.exit(1)
            logger.info(f"Downloading {len(products)} product(s)...")
            for prod in products:
                client.download_product(prod, extract=args.extract)

    elif args.command == "download-pair":
        if args.master_id and args.slave_id:
            res = client.download_pair(args.master_id, args.slave_id, extract=args.extract)
            logger.info(f"Pair downloaded:\n  Master: {res['master_path']}\n  Slave: {res['slave_path']}")
        else:
            products = client.search(
                start_date=args.start_date,
                end_date=args.end_date,
                bbox=bbox,
                aoi_wkt=args.aoi_wkt,
                track=args.track,
                orbit_direction=args.orbit_direction,
                top=args.top,
            )
            pairs = client.find_pairs(
                products,
                min_baseline_days=args.min_baseline,
                max_baseline_days=args.max_baseline,
            )
            if not pairs:
                logger.error("No compatible Master-Slave pair found to download.")
                sys.exit(1)

            selected_pair = pairs[0]
            logger.info(
                f"Selected optimal pair: Track {selected_pair['track']}, "
                f"Delta: {selected_pair['temporal_baseline_days']} days."
            )
            res = client.download_pair(
                selected_pair["master"],
                selected_pair["slave"],
                extract=args.extract,
            )
            logger.info(f"Pair downloaded:\n  Master: {res['master_path']}\n  Slave: {res['slave_path']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, PermissionError) as auth_err:
        logger.error(str(auth_err))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user.")
        sys.exit(130)
