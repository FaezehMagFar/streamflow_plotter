#!/usr/bin/env python3
"""
Streamflow Data Plotter for NWM and USGS Data

Retrieves streamflow data from two sources:
 - NOAA's National Water Model (NWM) v2.1 and v3 retrospective runs, read as
   individual hourly NetCDF files on S3 (NOT Zarr -- see note below)
 - USGS Water Services API

and plots them together for a given reach / gauge and time window.

WHY NETCDF INSTEAD OF ZARR:
NOAA publishes this dataset in both Zarr and plain NetCDF format. The Zarr
path is faster for large queries but depends on the `zarr` and `numcodecs`
packages, which are compiled C extensions that are notoriously prone to
version-mismatch errors on Windows (the "unrecognized engine 'zarr'" /
"cannot import name 'cbuffer_sizes'" family of errors). This version reads
the same data as one NetCDF file per hour instead, using only `h5netcdf`
and `netCDF4`, which are far more stable to install and are already part
of a standard xarray installation. The tradeoff is more, smaller network
requests instead of one lazy array -- fine for a single-reach, days-to-weeks
query like this one, but not a good choice if you need months of data
across many reaches.

Usage:
    python streamflow_plotter.py
    python streamflow_plotter.py --reach-id 18514402 --station-id 02378500 \
        --start 2020-08-24T00:00:00 --end 2020-09-03T23:59:59 \
        --site-name "Fish River near Silver Hill, AL" --output flood_event.png
"""

import argparse
import sys
from typing import List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import s3fs
import xarray as xr

CFS_TO_CMS = 0.0283168
USGS_PARAMETER_CODE = "00060"  # Discharge in cfs (use "72137" for tidally filtered discharge)

# NetCDF (not Zarr) buckets -- see module docstring.
NWM_V21_BUCKET = "noaa-nwm-retrospective-2-1-pds"
NWM_V21_PREFIX = "model_output"
NWM_V3_BUCKET = "noaa-nwm-retrospective-3-0-pds"
NWM_V3_PREFIX = "CONUS/netcdf/model_output"


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


def _nwm_netcdf_key(prefix: str, timestamp: pd.Timestamp) -> str:
    """Build the S3 key for a single hourly NWM retrospective CHRTOUT NetCDF file."""
    stamp = timestamp.strftime("%Y%m%d%H%M")
    year = timestamp.strftime("%Y")
    return f"{prefix}/{year}/{stamp}.CHRTOUT_DOMAIN1.comp"


def ReadNWMNetCDFData(
    bucket: str, prefix: str, reach_id: int, start_time: str, end_time: str, fs: s3fs.S3FileSystem = None
) -> Tuple[List[pd.Timestamp], List[float]]:
    """
    Read a single reach's hourly streamflow from NWM retrospective NetCDF files on S3.

    Opens one file per hour (each file covers the full CONUS domain at that hour)
    via the h5netcdf engine, which avoids the zarr/numcodecs dependency entirely.
    Missing hours are skipped with a warning rather than failing the whole run.
    """
    if fs is None:
        fs = s3fs.S3FileSystem(anon=True)

    hourly_times = pd.date_range(start=start_time, end=end_time, freq="h")
    if len(hourly_times) == 0:
        raise ValueError(f"No hourly timestamps between {start_time} and {end_time}")

    time_vals: List[pd.Timestamp] = []
    flow_vals: List[float] = []

    for i, ts in enumerate(hourly_times):
        key = f"{bucket}/{_nwm_netcdf_key(prefix, ts)}"
        try:
            with fs.open(key, "rb") as f:
                with xr.open_dataset(f, engine="h5netcdf") as ds:
                    flow = float(ds["streamflow"].sel(feature_id=reach_id).values)
        except FileNotFoundError:
            print(f"  warning: no file for {ts} ({key}), skipping")
            continue
        except KeyError:
            raise ValueError(f"Reach ID {reach_id} not found in {key}")

        time_vals.append(ts)
        flow_vals.append(flow)

        if (i + 1) % 24 == 0 or i == len(hourly_times) - 1:
            print(f"  ...{i + 1}/{len(hourly_times)} hours read from {bucket}")

    if not flow_vals:
        raise ValueError(f"No NWM data retrieved for reach {reach_id} from {bucket} between {start_time} and {end_time}")

    return time_vals, flow_vals


def AlignForPlotting(
    time_a: List[pd.Timestamp], flow_a: List[float], time_b: List[pd.Timestamp], flow_b: List[float]
) -> Tuple[List[pd.Timestamp], List[float], List[pd.Timestamp], List[float]]:
    """Truncate two equal-cadence NWM series to a common length for plotting."""
    min_length = min(len(flow_a), len(flow_b))
    return time_a[:min_length], flow_a[:min_length], time_b[:min_length], flow_b[:min_length]


def CreateStreamflowGraph(
    time_vals_v21,
    flow_vals_v21,
    time_vals_v3,
    flow_vals_v3,
    time_vals_usgs,
    flow_vals_usgs,
    reach_id: int,
    site_name: str,
    output_path: str = None,
) -> None:
    """
    Plot NWM v2.1, NWM v3, and USGS streamflow on one time series axis.
    NWM v3 is optional: pass time_vals_v3=None (or an empty sequence) to omit it,
    e.g. when the v3 retrospective archive doesn't have data for this window.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(time_vals_v21, flow_vals_v21, label="NWM v2.1 Streamflow", color="blue")
    if time_vals_v3 is not None and len(time_vals_v3) > 0:
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
    parser.add_argument(
        "--skip-v3", action="store_true",
        help="Skip NWM v3 entirely (useful if you already know it 404s for your date range/reach)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    usgs_start_date = args.start.split("T")[0]
    usgs_end_date = args.end.split("T")[0]

    fs = s3fs.S3FileSystem(anon=True)

    try:
        print("Fetching NWM v2.1 data (one file per hour, this will take a bit)...")
        time_v21, flow_v21 = ReadNWMNetCDFData(
            NWM_V21_BUCKET, NWM_V21_PREFIX, args.reach_id, args.start, args.end, fs=fs
        )

        time_v3, flow_v3 = [], []
        if not args.skip_v3:
            print("Fetching NWM v3 data...")
            try:
                time_v3, flow_v3 = ReadNWMNetCDFData(
                    NWM_V3_BUCKET, NWM_V3_PREFIX, args.reach_id, args.start, args.end, fs=fs
                )
            except ValueError as exc:
                # v3's retrospective netcdf archive doesn't always have data for every
                # reach/date range -- don't let that block the v2.1-vs-USGS comparison.
                print(f"  warning: NWM v3 unavailable, continuing without it ({exc})")
        else:
            print("Skipping NWM v3 (--skip-v3 set).")

        print("Fetching USGS data...")
        time_usgs, flow_usgs = ReadUSGSData(args.station_id, usgs_start_date, usgs_end_date)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if time_v3:
        time_v21_plot, flow_v21_plot, time_v3, flow_v3 = AlignForPlotting(time_v21, flow_v21, time_v3, flow_v3)
    else:
        time_v21_plot, flow_v21_plot = time_v21, flow_v21

    CreateStreamflowGraph(
        time_v21_plot, flow_v21_plot,
        time_v3, flow_v3,
        time_usgs, flow_usgs,
        reach_id=args.reach_id,
        site_name=args.site_name,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
