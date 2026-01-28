#!/usr/bin/env python3
"""
Performance Benchmark Script for Ad Performance Aggregator

Provides comprehensive performance monitoring including:
- Execution time
- Peak memory usage (RSS)
- Per-operation timing
- Comparison between different implementations

Usage:
    python benchmark.py --input ad_data.csv --output results/

Requirements:
    pip install psutil memory-profiler pandas
"""

import argparse
import os
import sys
import time
import tracemalloc
from typing import Dict, Any

# Import monitoring tools
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    print("Warning: psutil not installed. Memory monitoring will be limited.")
    print("Install with: pip install psutil")
    HAS_PSUTIL = False

try:
    from memory_profiler import memory_usage
    HAS_MEMORY_PROFILER = True
except ImportError:
    print("Warning: memory-profiler not installed.")
    print("Install with: pip install memory-profiler")
    HAS_MEMORY_PROFILER = False


class PerformanceMonitor:
    """Context manager for monitoring performance metrics."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = None

    def __enter__(self):
        self.start_time = time.time()

        if HAS_PSUTIL:
            self.process = psutil.Process(os.getpid())
            self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        if HAS_PSUTIL and self.process:
            self.end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            self.peak_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            self.memory_delta = self.end_memory - self.start_memory

    def get_stats(self) -> Dict[str, Any]:
        """Return performance statistics."""
        stats = {
            'operation': self.operation_name,
            'time_elapsed': self.elapsed_time,
        }

        if HAS_PSUTIL:
            stats.update({
                'start_memory_mb': self.start_memory,
                'end_memory_mb': self.end_memory,
                'peak_memory_mb': self.peak_memory,
                'memory_delta_mb': self.memory_delta,
            })

        return stats


def format_stats(stats: Dict[str, Any]) -> str:
    """Format statistics for display."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"PERFORMANCE: {stats['operation']}")
    lines.append('='*60)
    lines.append(f"Time Elapsed:     {stats['time_elapsed']:.2f} seconds")

    if 'peak_memory_mb' in stats:
        lines.append(f"Start Memory:    {stats['start_memory_mb']:.2f} MB")
        lines.append(f"End Memory:      {stats['end_memory_mb']:.2f} MB")
        lines.append(f"Peak Memory:     {stats['peak_memory_mb']:.2f} MB")
        lines.append(f"Memory Delta:    {stats['memory_delta_mb']:+.2f} MB")

    lines.append('='*60)
    return '\n'.join(lines)


def run_benchmark(input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
    """
    Run the aggregator with performance monitoring.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output directory
        **kwargs: Additional arguments for aggregator

    Returns:
        Dictionary with performance statistics
    """
    from aggregator import Aggregator

    print(f"\n{'='*60}")
    print("BENCHMARK: Ad Performance Aggregator")
    print('='*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Get file size
    if os.path.exists(input_path):
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"Size:   {file_size_mb:.2f} MB")

    print('='*60)

    # Initialize aggregator
    aggregator = Aggregator(
        progress_every=kwargs.get('progress_every', 5_000_000),
        chunksize=kwargs.get('chunksize', 1_000_000)
    )

    # Monitor processing
    with PerformanceMonitor("CSV Processing") as proc_monitor:
        accumulator = aggregator.process_file(
            input_path=input_path,
            delimiter=kwargs.get('delimiter', ','),
            encoding=kwargs.get('encoding', 'utf-8'),
            has_header=kwargs.get('has_header', True)
        )

    # Monitor ranking
    with PerformanceMonitor("Ranking (CTR + CPA)") as rank_monitor:
        top_10_ctr = aggregator.get_top_10_ctr(accumulator)
        top_10_cpa = aggregator.get_top_10_cpa(accumulator)

    # Monitor output
    with PerformanceMonitor("Output Generation") as output_monitor:
        ctr_output_path = os.path.join(output_path, 'top10_ctr.csv')
        cpa_output_path = os.path.join(output_path, 'top10_cpa.csv')

        os.makedirs(output_path, exist_ok=True)
        aggregator.write_output_csv(top_10_ctr, ctr_output_path)
        aggregator.write_output_csv(top_10_cpa, cpa_output_path)

    # Compile results
    results = {
        'processing': proc_monitor.get_stats(),
        'ranking': rank_monitor.get_stats(),
        'output': output_monitor.get_stats(),
        'total_time': proc_monitor.elapsed_time + rank_monitor.elapsed_time + output_monitor.elapsed_time,
        'rows_processed': aggregator.stats['rows_read'],
        'unique_campaigns': aggregator.stats['unique_campaigns'],
        'rows_skipped': aggregator.stats['rows_skipped'],
        'parse_errors': aggregator.stats['parse_errors'],
    }

    # Print detailed stats
    print(format_stats(results['processing']))
    print(format_stats(results['ranking']))
    print(format_stats(results['output']))

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print('='*60)
    print(f"Total Time:        {results['total_time']:.2f} seconds")
    print(f"Rows Processed:    {results['rows_processed']:,}")
    print(f"Unique Campaigns:  {results['unique_campaigns']:,}")
    print(f"Rows Skipped:      {results['rows_skipped']:,}")
    print(f"Parse Errors:      {results['parse_errors']:,}")

    if results['rows_processed'] > 0:
        throughput = results['rows_processed'] / results['processing']['time_elapsed']
        print(f"Throughput:        {throughput:,.0f} rows/second")

    if 'peak_memory_mb' in results['processing']:
        print(f"Peak Memory:       {results['processing']['peak_memory_mb']:.2f} MB")

    if os.path.exists(input_path):
        print(f"File Size:         {file_size_mb:.2f} MB")
        print(f"Processing Rate:   {file_size_mb / results['processing']['time_elapsed']:.2f} MB/second")

    print('='*60)

    return results


def run_memory_profiler(input_path: str, output_path: str, **kwargs):
    """
    Run detailed memory profiling using memory_profiler.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output directory
        **kwargs: Additional arguments for aggregator
    """
    if not HAS_MEMORY_PROFILER:
        print("Error: memory-profiler is required for detailed profiling.")
        print("Install with: pip install memory-profiler")
        sys.exit(1)

    from aggregator import Aggregator

    print(f"\n{'='*60}")
    print("MEMORY PROFILER: Detailed Analysis")
    print('='*60)

    def run_aggregator():
        aggregator = Aggregator(
            progress_every=kwargs.get('progress_every', 5_000_000),
            chunksize=kwargs.get('chunksize', 1_000_000)
        )

        accumulator = aggregator.process_file(
            input_path=input_path,
            delimiter=kwargs.get('delimiter', ','),
            encoding=kwargs.get('encoding', 'utf-8'),
            has_header=kwargs.get('has_header', True)
        )

        top_10_ctr = aggregator.get_top_10_ctr(accumulator)
        top_10_cpa = aggregator.get_top_10_cpa(accumulator)

        os.makedirs(output_path, exist_ok=True)
        aggregator.write_output_csv(
            top_10_ctr,
            os.path.join(output_path, 'top10_ctr.csv')
        )
        aggregator.write_output_csv(
            top_10_cpa,
            os.path.join(output_path, 'top10_cpa.csv')
        )

        return aggregator.stats

    # Run with memory profiler
    # Returns (mem_usage_list, retval) when retval=True
    mem_usage, stats = memory_usage(
        (run_aggregator,),
        interval=0.1,
        timeout=None,
        retval=True
    )

    print(f"\n{'='*60}")
    print("MEMORY PROFILER RESULTS")
    print('='*60)
    print(f"Rows Processed:    {stats['rows_read']:,}")
    print(f"Unique Campaigns:  {stats['unique_campaigns']:,}")
    print(f"Elapsed Time:      {stats['elapsed_seconds']:.2f} seconds")
    print(f"\nMemory Usage:")
    print(f"  Min:  {min(mem_usage):.2f} MiB")
    print(f"  Max:  {max(mem_usage):.2f} MiB")
    print(f"  Avg:  {sum(mem_usage) / len(mem_usage):.2f} MiB")
    print('='*60)


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description='Benchmark Ad Performance Aggregator with performance monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmark.py --input ad_data.csv --output results/

  # Detailed memory profiling
  python benchmark.py --input ad_data.csv --output results/ --profile-memory

  # Custom chunk size
  python benchmark.py --input ad_data.csv --output results/ --chunksize 500000

  # Compare different chunk sizes
  python benchmark.py --input ad_data.csv --output results1/ --chunksize 500000
  python benchmark.py --input ad_data.csv --output results2/ --chunksize 2000000
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to output directory'
    )

    parser.add_argument(
        '--profile-memory',
        action='store_true',
        help='Run detailed memory profiling (requires memory-profiler)'
    )

    parser.add_argument(
        '--delimiter',
        default=',',
        help='CSV delimiter character (default: ",")'
    )

    parser.add_argument(
        '--encoding',
        default='utf-8',
        help='File encoding (default: utf-8)'
    )

    parser.add_argument(
        '--has-header',
        type=lambda x: x.lower() in ('true', '1', 'yes', 'on'),
        default=True,
        help='Whether CSV has a header row (default: true)'
    )

    parser.add_argument(
        '--progress-every',
        type=int,
        default=5_000_000,
        help='Log progress every N rows (default: 5_000_000)'
    )

    parser.add_argument(
        '--chunksize',
        type=int,
        default=1_000_000,
        help='Number of rows to process per chunk (default: 1_000_000)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run appropriate benchmark
    if args.profile_memory:
        run_memory_profiler(
            input_path=args.input,
            output_path=args.output,
            delimiter=args.delimiter,
            encoding=args.encoding,
            has_header=args.has_header,
            progress_every=args.progress_every,
            chunksize=args.chunksize
        )
    else:
        results = run_benchmark(
            input_path=args.input,
            output_path=args.output,
            delimiter=args.delimiter,
            encoding=args.encoding,
            has_header=args.has_header,
            progress_every=args.progress_every,
            chunksize=args.chunksize
        )

        # Return success if no errors
        sys.exit(0 if results['parse_errors'] == 0 else 1)


if __name__ == '__main__':
    main()
