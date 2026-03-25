#!/usr/bin/env python

import argparse
import re
import time
from datetime import datetime

import astropy.units as u
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tqdm import tqdm

from sotrplib.solar_system.solar_system import load_mpc_orbital_database

"""
This script queries JPL Horizons for asteroid ephemerides and saves the results to a Parquet file.
Initially designed to run in batches, but due to API limitations, it currently queries one asteroid at a time.

This may take several hours to run for a large number of asteroids and several year timerange.

It is a bit outdated in the way that I grab the asteroids because I'm using the MPC orbital database...
Can fix this later.

AF 25 Mar 2026

"""


HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons_file.api"

SITE_COORD = "-67.7876,-22.9585,5.2"


def chunk(lst, n):
    """Yield successive n-sized chunks."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_payload(designations, start, stop):
    commands = "".join([f"COMMAND='{d}'" for d in designations])
    payload = f"""
!$$SOF
{commands}
CENTER='coord'
SITE_COORD='{SITE_COORD}'
MAKE_EPHEM='YES'
TABLE_TYPE='OBSERVER'
START_TIME={start}
STOP_TIME={stop}
STEP_SIZE='2h'
QUANTITIES='1,3,20'
CSV_FORMAT='YES'
"""

    return payload


def query_batch(designations, start, stop, max_retries=5):
    payload = build_payload(designations, start, stop)

    for attempt in range(max_retries):
        try:
            r = requests.post(
                HORIZONS_URL,
                data={"format": "text", "input": payload},
                timeout=120,  # avoid hanging
            )
            r.raise_for_status()
            return r.text

        except requests.exceptions.HTTPError as e:
            if r.status_code == 503:
                wait = 5 * (2**attempt)  # exponential backoff
                print(f"Horizons 503, retrying in {wait}s (attempt {attempt + 1})...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.RequestException as e:
            wait = 5 * (2**attempt)
            print(f"Request error: {e}, retrying in {wait}s (attempt {attempt + 1})...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_batch_output(text, designations):
    rows = []

    current_obj = None
    inside = False

    for line in text.splitlines():
        if "Target body name" in line:
            m = re.search(r"Target body name:\s*(.+?)\s*\(", line)
            if m:
                current_obj = m.group(1)

        elif "$$SOE" in line:
            inside = True

        elif "$$EOE" in line:
            inside = False

        elif inside:
            parts = line.split(",")

            if len(parts) < 4:
                continue

            dt_utc = str(parts[0]).lstrip()
            ra_str = str(parts[3])
            dec_str = str(parts[4])
            coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
            dt_utc = datetime.strptime(dt_utc, "%Y-%b-%d %H:%M")

            # Convert datetime to Julian Date
            jd = Time(dt_utc, scale="utc").jd
            delta = float(parts[8])

            rows.append(
                {
                    "designation": current_obj,
                    "datetime_utc": dt_utc,
                    "julian_day": jd,
                    "ra_deg": coord.ra.to_value(u.deg),
                    "dec_deg": coord.dec.to_value(u.deg),
                    "distance_au": delta,
                }
            )

    return pd.DataFrame(rows)


def main(start, stop):
    print("Querying JPL Horizons for asteroid ephemerides between:")
    print(start, " and ", stop)
    orbital_db = load_mpc_orbital_database("mpc_orbital_params_bright_asteroids.csv")

    ## getting asteroid designations used by horizons api
    ## that is number and semicolon to indicate small body
    ## because why not
    designations = [
        s.split(")")[0].strip("(") + ";" for s in orbital_db["designation"].tolist()
    ]
    print(len(designations), "total asteroids")
    outfile = f"JPL_batched_ephemerides_{start}_{stop}.parquet"

    writer = None
    ## cant do batch submissions to horizons so just one at a time.
    for batch in tqdm(list(chunk(designations, 1))):
        text = query_batch(batch, start, stop)
        print(text.count("Target body name"))
        df = parse_batch_output(text, batch)
        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(outfile, table.schema)

        writer.write_table(table)

    if writer is not None:
        writer.close()

    print(f"Finished {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--stop", type=str, default="2035-01-01")

    args = parser.parse_args()

    main(args.start, args.stop)
