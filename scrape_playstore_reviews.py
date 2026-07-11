#!/usr/bin/env python3
"""
Scrape Google Play Store reviews for the Wondr by BNI application.

The script uses google-play-scraper to paginate through review batches until the
target number of rows is collected. Results are saved as a CSV file that matches
the dataset format used in the sentiment analysis notebook.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from google_play_scraper import Sort, reviews


DEFAULT_APP_ID = "com.bni.wonder"
DEFAULT_MIN_ROWS = 3000
DEFAULT_BATCH_SIZE = 200
DEFAULT_SLEEP_SEC = 0.8


@dataclass
class ScrapeConfig:
    app_id: str = DEFAULT_APP_ID
    lang: str = "id"
    country: str = "id"
    min_rows: int = DEFAULT_MIN_ROWS
    batch_size: int = DEFAULT_BATCH_SIZE
    sleep_sec: float = DEFAULT_SLEEP_SEC
    sort_order: Sort = Sort.NEWEST


def _iter_reviews(cfg: ScrapeConfig) -> Iterable[dict]:
    """Yield review dicts until min_rows is reached or no more data is available."""
    collected = 0
    token = None

    while True:
        batch, token = reviews(
            cfg.app_id,
            lang=cfg.lang,
            country=cfg.country,
            sort=cfg.sort_order,
            count=cfg.batch_size,
            continuation_token=token,
        )
        if not batch:
            break

        for item in batch:
            yield item
            collected += 1

        if collected >= cfg.min_rows:
            break

        if token is None:
            break

        time.sleep(cfg.sleep_sec)


def scrape(cfg: ScrapeConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for idx, row in enumerate(_iter_reviews(cfg), start=1):
        rows.append(row)
        if idx % 500 == 0:
            print(f"Collected {idx} reviews…", file=sys.stderr)

    if not rows:
        raise RuntimeError("Scraper did not return any reviews. Check app_id/lang/country.")

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset="reviewId", inplace=True)
    df.sort_values(by="at", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app-id",
        default=DEFAULT_APP_ID,
        help="Google Play application id (default: %(default)s)",
    )
    parser.add_argument(
        "--lang",
        default="id",
        help="Language locale for reviews (default: %(default)s)",
    )
    parser.add_argument(
        "--country",
        default="id",
        help="Country locale for reviews (default: %(default)s)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help="Minimum number of rows to collect (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of reviews per API request (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=DEFAULT_SLEEP_SEC,
        help="Delay between API calls in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wondr_reviews_gps.csv"),
        help="Path to write the resulting CSV (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ScrapeConfig(
        app_id=args.app_id,
        lang=args.lang,
        country=args.country,
        min_rows=args.min_rows,
        batch_size=args.batch_size,
        sleep_sec=args.sleep_sec,
    )

    print(
        f"Scraping Google Play reviews for {cfg.app_id} "
        f"({cfg.lang}-{cfg.country}) until {cfg.min_rows} rows…",
        file=sys.stderr,
    )
    df = scrape(cfg)
    print(f"Total unique reviews collected: {len(df):,}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved dataset to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
