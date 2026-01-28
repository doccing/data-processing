#!/usr/bin/env python3
"""
Ad Performance Aggregator

Optimizations applied:
1. Vectorized accumulator updates (itertuples instead of iterrows)
2. Single-pass numeric conversion with parse error detection
3. Vectorized array operations for result building
4. Efficient ranking without DataFrame reconstruction
5. Optimized CSV output with buffering
6. Reduced function call overhead in hot paths

Target: 500K+ rows/second in Docker
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

# Configure logging to stderr for progress updates
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class Aggregator:
    """
    Aggregator using vectorized operations.

    Key optimizations:
    - Uses groupby().sum() with itertuples() for fast dict updates
    - Pre-allocates numpy arrays for fast aggregation
    - Avoids iterrows() (uses itertuples() and vectorized operations instead)
    - Single-pass numeric conversion with parse error detection
    - Vectorized array operations for result building
    """

    def __init__(self, progress_every: int = 5_000_000, chunksize: int = 1_000_000):
        """
        Initialize the aggregator.

        Args:
            progress_every: Log progress every N rows
            chunksize: Number of rows to process per chunk
        """
        self.progress_every = progress_every
        self.chunksize = chunksize
        self.stats = {
            'rows_read': 0,
            'rows_skipped': 0,
            'parse_errors': 0,
        }

    def process_file(
        self,
        input_path: str,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        has_header: bool = True
    ) -> Dict[str, Dict]:
        """
        Process the entire CSV file with vectorized aggregation.

        Args:
            input_path: Path to input CSV file
            delimiter: CSV delimiter character
            encoding: File encoding
            has_header: Whether the CSV has a header row

        Returns:
            Dictionary mapping campaign_id to aggregated stats
        """
        start_time = time.time()

        # Use defaultdict for fast accumulator updates
        accumulator = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'spend': 0.0,
            'conversions': 0
        })

        rows_read = 0

        # Read CSV in chunks
        reader = pd.read_csv(
            input_path,
            sep=delimiter,
            encoding=encoding,
            header=0 if has_header else None,
            names=['campaign_id', 'date', 'impressions', 'clicks', 'spend', 'conversions'] if not has_header else None,
            chunksize=self.chunksize,
            low_memory=False,
            engine='c',
            na_values=['', 'NA', 'N/A', 'null', 'NULL'],
            keep_default_na=False,
            on_bad_lines='warn'
        )

        for chunk in reader:
            chunk_rows = len(chunk)
            rows_read += chunk_rows

            # Progress logging
            if rows_read % self.progress_every < chunk_rows:
                logger.info(f"Processed {rows_read:,} rows...")

            # Strip whitespace from campaign_id BEFORE checking for empty
            if 'campaign_id' in chunk.columns:
                chunk['campaign_id'] = chunk['campaign_id'].astype(str).str.strip()

            # Track parse errors and convert to numeric in one pass (TRULY VECTORIZED)
            parse_error_mask = pd.Series([False] * len(chunk), index=chunk.index)

            for col in ['impressions', 'clicks', 'spend', 'conversions']:
                if col not in chunk.columns:
                    continue

                # Store original values for parse error detection
                original_values = chunk[col].copy()

                # Convert to numeric (coerce errors to NaN)
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

                # Detect parse errors: values that were strings but failed to convert
                if ptypes.is_string_dtype(original_values):
                    # Vectorized check: find non-empty strings
                    is_non_empty_string = original_values.astype(str).str.strip().ne('') & (original_values != 'nan')
                    # Parse errors where: was a non-empty string AND conversion resulted in NaN
                    parse_error_mask |= (is_non_empty_string & chunk[col].isna())

            # Count parse errors (rows with non-numeric strings)
            self.stats['parse_errors'] += int(parse_error_mask.sum())

            # Filter invalid rows (vectorized)
            valid_mask = (
                chunk['campaign_id'].notna() &
                (chunk['campaign_id'] != '') &
                chunk['impressions'].notna() &
                chunk['clicks'].notna() &
                chunk['spend'].notna() &
                chunk['conversions'].notna() &
                (chunk['impressions'] >= 0) &
                (chunk['clicks'] >= 0) &
                (chunk['spend'] >= 0) &
                (chunk['conversions'] >= 0)
            )

            # Count rows skipped (rows that don't pass validation)
            self.stats['rows_skipped'] += int((~valid_mask).sum())

            chunk_valid = chunk[valid_mask].copy()

            if len(chunk_valid) == 0:
                continue

            # Vectorized aggregation (ULTRA FAST)
            # Group by campaign_id and sum all numeric columns at once
            grouped = chunk_valid.groupby('campaign_id', observed=True)[
                ['impressions', 'clicks', 'spend', 'conversions']
            ].sum()

            # Fast dict update - use itertuples() instead of iterrows() (10-20x faster)
            for row in grouped.itertuples():
                campaign_id = row.Index
                acc = accumulator[campaign_id]
                acc['impressions'] += int(row.impressions)
                acc['clicks'] += int(row.clicks)
                acc['spend'] += float(row.spend)
                acc['conversions'] += int(row.conversions)

        # Final stats
        elapsed = time.time() - start_time
        self.stats['rows_read'] = rows_read
        self.stats['elapsed_seconds'] = elapsed
        self.stats['unique_campaigns'] = len(accumulator)

        return dict(accumulator)

    def get_top_10_ctr(self, accumulator: Dict[str, Dict]) -> List[Tuple]:
        """
        Get top 10 campaigns by highest CTR.
        """
        # Build lists for vectorized sorting
        campaigns = []
        for cid, stats in accumulator.items():
            impressions = stats['impressions']
            clicks = stats['clicks']
            ctr = clicks / impressions if impressions > 0 else 0.0

            campaigns.append({
                'campaign_id': cid,
                'impressions': impressions,
                'clicks': clicks,
                'spend': stats['spend'],
                'conversions': stats['conversions'],
                'CTR': ctr
            })

        if not campaigns:
            return []

        # Convert to DataFrame for fast sorting
        df = pd.DataFrame(campaigns)

        # Single sort with all tie-breakers
        df_sorted = df.sort_values(
            by=['CTR', 'clicks', 'impressions', 'campaign_id'],
            ascending=[False, False, False, True]
        )

        # Extract top 10 as list of tuples (vectorized)
        top_10 = df_sorted.head(10)

        # Vectorized CPA calculation
        conversions_array = top_10['conversions'].values
        spend_array = top_10['spend'].values
        cpa_array = np.divide(
            spend_array,
            conversions_array,
            out=np.full_like(spend_array, np.nan, dtype=float),
            where=conversions_array > 0
        )

        # Build result using zip on arrays (much faster than iterrows)
        result = [
            (
                cid,
                int(imp),
                int(clk),
                float(spd),
                int(conv),
                float(ctr),
                float(cpa) if not np.isnan(cpa) else None
            )
            for cid, imp, clk, spd, conv, ctr, cpa in zip(
                top_10['campaign_id'].values,
                top_10['impressions'].values,
                top_10['clicks'].values,
                top_10['spend'].values,
                top_10['conversions'].values,
                top_10['CTR'].values,
                cpa_array
            )
        ]

        return result

    def get_top_10_cpa(self, accumulator: Dict[str, Dict]) -> List[Tuple]:
        """
        Get top 10 campaigns by lowest CPA.
        """
        campaigns = []
        for cid, stats in accumulator.items():
            conversions = stats['conversions']

            # Skip zero conversions
            if conversions <= 0:
                continue

            impressions = stats['impressions']
            clicks = stats['clicks']
            spend = stats['spend']

            cpa = spend / conversions
            ctr = clicks / impressions if impressions > 0 else 0.0

            campaigns.append({
                'campaign_id': cid,
                'impressions': impressions,
                'clicks': clicks,
                'spend': spend,
                'conversions': conversions,
                'CPA': cpa,
                'CTR': ctr
            })

        if not campaigns:
            return []

        df = pd.DataFrame(campaigns)

        # Single sort with all tie-breakers
        df_sorted = df.sort_values(
            by=['CPA', 'conversions', 'spend', 'campaign_id'],
            ascending=[True, False, True, True]
        )

        top_10 = df_sorted.head(10)

        # Build result using zip on arrays (much faster than iterrows)
        result = [
            (
                cid,
                int(imp),
                int(clk),
                float(spd),
                int(conv),
                float(ctr),
                float(cpa)
            )
            for cid, imp, clk, spd, conv, ctr, cpa in zip(
                top_10['campaign_id'].values,
                top_10['impressions'].values,
                top_10['clicks'].values,
                top_10['spend'].values,
                top_10['conversions'].values,
                top_10['CTR'].values,
                top_10['CPA'].values
            )
        ]

        return result

    def write_output_csv(self, campaigns: List[Tuple], output_path: str) -> None:
        """
        Write results to CSV file (optimized with buffering).
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Use write() with buffering for speed
        with open(output_path, 'w', newline='', encoding='utf-8', buffering=8192) as f:
            # Write header
            f.write('campaign_id,total_impressions,total_clicks,total_spend,total_conversions,CTR,CPA\n')

            # Write rows (optimized formatting)
            for campaign_id, impressions, clicks, spend, conversions, ctr, cpa in campaigns:
                # Pre-format all values
                spend_str = f"{spend:.2f}"
                ctr_str = f"{ctr:.4f}"
                cpa_str = f"{cpa:.2f}" if cpa is not None else ''

                # Single write call per row (faster)
                f.write(f"{campaign_id},{impressions},{clicks},{spend_str},{conversions},{ctr_str},{cpa_str}\n")

    def print_stats(self) -> None:
        """Print aggregation statistics to stderr."""
        logger.info("\n" + "="*60)
        logger.info("AGGREGATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Rows read:        {self.stats['rows_read']:,}")
        logger.info(f"Rows skipped:     {self.stats['rows_skipped']:,}")
        logger.info(f"Parse errors:     {self.stats['parse_errors']:,}")
        logger.info(f"Unique campaigns: {self.stats['unique_campaigns']:,}")
        logger.info(f"Elapsed time:     {self.stats['elapsed_seconds']:.2f} seconds")
        if self.stats['rows_read'] > 0:
            logger.info(f"Throughput:       {self.stats['rows_read'] / self.stats['elapsed_seconds']:,.0f} rows/second")
        logger.info("="*60)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Aggregate advertising performance data by campaign',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregator.py --input ad_data.csv --output results/
  python aggregator.py -i ad_data.csv -o results/ --chunksize 2000000

Optimizations:
  - Vectorized accumulator updates (itertuples not iterrows)
  - Single-pass numeric conversion
  - Efficient ranking algorithms
  - Buffered I/O for output

Expected throughput: 500K+ rows/second (native), 400K+ rows/second (Docker)
        """
    )

    parser.add_argument('-i', '--input', required=True, help='Path to input CSV file')
    parser.add_argument('-o', '--output', required=True, help='Path to output directory')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter character (default: ",")')
    parser.add_argument('--encoding', default='utf-8', help='File encoding (default: utf-8)')
    parser.add_argument('--has-header', type=lambda x: x.lower() in ('true', '1', 'yes', 'on'), default=True, help='Whether CSV has a header row (default: true)')
    parser.add_argument('--progress-every', type=int, default=5_000_000, help='Log progress every N rows (default: 5_000_000)')
    parser.add_argument('--chunksize', type=int, default=1_000_000, help='Number of rows to process per chunk (default: 1_000_000). Higher = faster but more memory.')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.isfile(args.input):
        logger.error(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Initialize aggregator
    aggregator = Aggregator(
        progress_every=args.progress_every,
        chunksize=args.chunksize
    )

    # Process file
    logger.info(f"Processing: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Chunk size: {args.chunksize:,} rows")
    logger.info("")

    accumulator = aggregator.process_file(
        input_path=args.input,
        delimiter=args.delimiter,
        encoding=args.encoding,
        has_header=args.has_header
    )

    # Generate outputs
    logger.info("\nGenerating output files...")

    top_10_ctr = aggregator.get_top_10_ctr(accumulator)
    ctr_output_path = os.path.join(args.output, 'top10_ctr.csv')
    aggregator.write_output_csv(top_10_ctr, ctr_output_path)
    logger.info(f"Written: {ctr_output_path}")

    top_10_cpa = aggregator.get_top_10_cpa(accumulator)
    cpa_output_path = os.path.join(args.output, 'top10_cpa.csv')
    aggregator.write_output_csv(top_10_cpa, cpa_output_path)
    logger.info(f"Written: {cpa_output_path}")

    # Print statistics
    aggregator.print_stats()

    logger.info(f"\n✓ Successfully processed {aggregator.stats['rows_read']:,} rows")
    logger.info(f"✓ Output files written to: {args.output}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
