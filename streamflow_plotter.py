#!/usr/bin/env python3
"""
Streamflow Data Plotter for NWM and USGS Data

Retrieves streamflow data from two sources:
 - NOAA's National Water Model (NWM) v2.1 and v3 retrospective runs (AWS, Zarr)
 - USGS Water Services API

and plots them together for a given reach / gauge and time window.

Usage:
    python streamflow_plotter.py
    python streamflow_plotter.py --reach-id 18514402 --station-id 02378500 \
        --start 2020-08-24T00:00:00 --end 2020-09-03T23:59:59 \
        --site-name "Fish River near Silver Hill, AL" --output flood_event.png
"""

import argparse
import sys
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import s3fs
import xarray as xr

CFS_TO_CMS = 0.0283168
USGS_PARAMETER_CODE = "00060"  # Discharge in cfs (use "72137" for tidally filtered discharge)

NWM_V21_URL = "s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr"
NWM_V3_URL = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
NWM_ZERO_TIME = np.datetime64("1979-02-01T00:00:00")


def ReadUSGSData(station_id: str, start_date: str, end_date: str) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Fetch instantaneous discharge from the USGS Water Services API and convert cfs -> cms."""
    url = (
        "https://waterservices.usgs.gov/nwis/iv/"
        f"?format=json&sites={station_id}&startDT={start_date}&endDT={end_date}"
        f"&parameterCd={USGS_PARAMETER_CODE}"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"USGS request failed for station {station_id}: {exc}") from exc

    data = response.json()
    series = data.get("value", {}).get("timeSeries", [])
    if not series:
        raise ValueError(f"No USGS data available for station {station_id} from {start_date} to {end_date}")

    values = series[0]["values"][0]["value"]
    time_vals = pd.to_datetime([v["dateTime"] for v in values])
    flow_vals = np.array([float(v["value"]) for v in values]) * CFS_TO_CMS
    return time_vals, flow_vals


def ReadNWMZarrData(
    zarr_url: str, reach_id: int, start_time: str, end_time: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a single reach's streamflow time series from an NWM retrospective Zarr store on S3.

    Only the requested time window and a single feature index are pulled from S3 (via isel),
    rather than evaluating a boolean mask across the full multi-decade array.
    """
    fs = s3fs.S3FileSystem(anon=True)
    store = xr.open_zarr(s3fs.S3Map(zarr_url, s3=fs), consolidated=False)

    start_idx = int((np.datetime64(start_time) - NWM_ZERO_TIME) / np.timedelta64(1, "h"))
    end_idx = int((np.datetime64(end_time) - NWM_ZERO_TIME) / np.timedelta64(1, "h"))

    # Load only the (small) feature_id coordinate to locate the reach, not the full dataset.
    feature_ids = store["feature_id"].values
    matches = np.where(feature_ids == reach_id)[0]
    if matches.size == 0:
        raise ValueError(f"Reach ID {reach_id} not found in {zarr_url}")
    feature_idx = int(matches[0])

    subset = store["streamflow"].isel(time=slice(start_idx, end_idx), feature_id=feature_idx)
    flow_vals = subset.values
    time_vals = store["time"].isel(time=slice(start_idx, end_idx)).values
    return time_vals, flow_vals


def AlignForPlotting(
    time_a: np.ndarray, flow_a: np.ndarray, time_b: np.ndarray, flow_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truncate two equal-cadence NWM series to a common length for plotting."""
    min_length = min(len(flow_a), len(flow_b))
    return time_a[:min_length], flow_a[:min_length], time_b[:min_length], flow_b[:min_length]


def CreateStreamflowGraph(
    time_vals_v21: np.ndarray,
    flow_vals_v21: np.ndarray,
    time_vals_v3: np.ndarray,
    flow_vals_v3: np.ndarray,
    time_vals_usgs: pd.DatetimeIndex,
    flow_vals_usgs: np.ndarray,
    reach_id: int,
    site_name: str,
    output_path: str = None,
) -> None:
    """Plot NWM v2.1, NWM v3, and USGS streamflow on one time series axis."""
    plt.figure(figsize=(10, 6))

    plt.plot(time_vals_v21, flow_vals_v21, label="NWM v2.1 Streamflow", color="blue")
    plt.plot(time_vals_v3, flow_vals_v3, label="NWM v3 Streamflow", color="red")
    plt.plot(time_vals_usgs, flow_vals_usgs, label="USGS Streamflow", color="black", linestyle="--")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    plt.xlabel("Date (Daily)")
    plt.ylabel("Streamflow (cms)")
    plt.title(f"Daily Streamflow at {site_name} (NWMID#{reach_id})")
    plt.legend(loc="upper right")

    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NWM v2.1/v3 and USGS streamflow for a reach/gauge.")
    parser.add_argument("--reach-id", type=int, default=18514402, help="NWM reach ID (default: 18514402)")
    parser.add_argument("--station-id", type=str, default="02378500", help="USGS station ID (default: 02378500)")
    parser.add_argument("--start", type=str, default="2020-08-24T00:00:00", help="Start time, ISO format")
    parser.add_argument("--end", type=str, default="2020-09-03T23:59:59", help="End time, ISO format")
    parser.add_argument(
        "--site-name", type=str, default="Fish River near Silver Hill, AL", help="Site name for plot title"
    )
    parser.add_argument("--output", type=str, default=None, help="If set, save figure here instead of showing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    usgs_start_date = args.start.split("T")[0]
    usgs_end_date = args.end.split("T")[0]

    try:
        print("Fetching NWM v2.1 data...")
        time_v21, flow_v21 = ReadNWMZarrData(NWM_V21_URL, args.reach_id, args.start, args.end)

        print("Fetching NWM v3 data...")
        time_v3, flow_v3 = ReadNWMZarrData(NWM_V3_URL, args.reach_id, args.start, args.end)

        print("Fetching USGS data...")
        time_usgs, flow_usgs = ReadUSGSData(args.station_id, usgs_start_date, usgs_end_date)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    time_v21, flow_v21, time_v3, flow_v3 = AlignForPlotting(time_v21, flow_v21, time_v3, flow_v3)

    CreateStreamflowGraph(
        time_v21, flow_v21,
        time_v3, flow_v3,
        time_usgs, flow_usgs,
        reach_id=args.reach_id,
        site_name=args.site_name,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
