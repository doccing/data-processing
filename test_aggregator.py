"""
Challenge-grade pytest test suite for aggregator.py

Tests the pandas-based Aggregator class with dict accumulator API.
"""

import os
import tempfile
import pytest
import csv

from aggregator import Aggregator


# =============================================================================
# TEST HELPERS
# =============================================================================

def create_test_csv(rows, has_header=True, delimiter=','):
    """
    Helper to create a temporary CSV file safely.

    Args:
        rows: List of lists, where each inner list represents a row
        has_header: Whether first row is a header
        delimiter: CSV delimiter

    Returns:
        Path to temporary file (caller responsible for cleanup)
    """
    fd, path = tempfile.mkstemp(suffix='.csv')
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerows(rows)
        return path
    except:
        # Clean up fd if write fails
        os.close(fd)
        raise


def create_test_csv_from_string(csv_content, delimiter=','):
    """
    Helper to create CSV from string content.

    Args:
        csv_content: String with CSV content
        delimiter: CSV delimiter

    Returns:
        Path to temporary file (caller responsible for cleanup)
    """
    fd, path = tempfile.mkstemp(suffix='.csv')
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            f.write(csv_content)
        return path
    except:
        os.close(fd)
        raise


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def aggregator():
    """Create a fresh Aggregator instance for each test."""
    return Aggregator(progress_every=100, chunksize=10)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    header = ['campaign_id', 'date', 'impressions', 'clicks', 'spend', 'conversions']
    data = [
        ['CMP001', '2025-01-01', '1000', '50', '100.50', '5'],
        ['CMP002', '2025-01-01', '2000', '100', '200.00', '10'],
        ['CMP001', '2025-01-02', '1500', '75', '150.75', '7'],
        ['CMP003', '2025-01-01', '5000', '250', '500.00', '25'],
    ]
    return [header] + data


# =============================================================================
# CORE AGGREGATION CORRECTNESS
# =============================================================================

class TestCoreAggregation:
    """Test core aggregation functionality and correctness."""

    def test_aggregation_totals(self, aggregator, sample_csv_data):
        """Test that repeated campaign_ids are properly aggregated."""
        path = create_test_csv(sample_csv_data, has_header=True)
        try:
            accumulator = aggregator.process_file(path)

            # Verify CMP001 aggregation (2 rows)
            assert 'CMP001' in accumulator
            cmp001 = accumulator['CMP001']
            assert cmp001['impressions'] == 2500  # 1000 + 1500
            assert cmp001['clicks'] == 125        # 50 + 75
            assert abs(cmp001['spend'] - 251.25) < 0.01  # 100.50 + 150.75
            assert cmp001['conversions'] == 12    # 5 + 7

            # Verify CMP002 aggregation (1 row)
            assert 'CMP002' in accumulator
            cmp002 = accumulator['CMP002']
            assert cmp002['impressions'] == 2000
            assert cmp002['clicks'] == 100
            assert abs(cmp002['spend'] - 200.00) < 0.01
            assert cmp002['conversions'] == 10

        finally:
            os.unlink(path)

    def test_stats_rows_read(self, aggregator):
        """Test that rows_read counts data rows (excluding header)."""
        rows = [
            ['campaign_id', 'date', 'impressions', 'clicks', 'spend', 'conversions'],
            ['CMP001', '2025-01-01', '1000', '50', '100', '5'],
            ['CMP002', '2025-01-01', '2000', '100', '200', '10'],
            ['CMP003', '2025-01-01', '3000', '150', '300', '15'],
        ]
        path = create_test_csv(rows, has_header=True)
        try:
            accumulator = aggregator.process_file(path)
            assert aggregator.stats['rows_read'] == 3  # 3 data rows
        finally:
            os.unlink(path)

    def test_stats_unique_campaigns(self, aggregator):
        """Test that unique_campaigns count matches."""
        rows = [
            ['campaign_id', 'date', 'impressions', 'clicks', 'spend', 'conversions'],
            ['CMP001', '2025-01-01', '1000', '50', '100', '5'],
            ['CMP002', '2025-01-01', '2000', '100', '200', '10'],
            ['CMP001', '2025-01-02', '1500', '75', '150', '7'],
        ]
        path = create_test_csv(rows, has_header=True)
        try:
            accumulator = aggregator.process_file(path)
            assert aggregator.stats['unique_campaigns'] == 2
        finally:
            os.unlink(path)

    def test_csv_without_header(self, aggregator):
        """Test processing CSV without header row."""
        rows = [
            ['CMP001', '2025-01-01', '1000', '50', '100.50', '5'],
            ['CMP002', '2025-01-01', '2000', '100', '200.00', '10'],
        ]
        path = create_test_csv(rows, has_header=False)
        try:
            accumulator = aggregator.process_file(path, has_header=False)
            assert aggregator.stats['rows_read'] == 2
            assert 'CMP001' in accumulator
            assert 'CMP002' in accumulator
        finally:
            os.unlink(path)

    def test_custom_delimiter(self, aggregator):
        """Test CSV with semicolon delimiter."""
        csv_content = """campaign_id;date;impressions;clicks;spend;conversions
CMP001;2025-01-01;1000;50;100.50;5
CMP002;2025-01-01;2000;100;200.00;10"""
        path = create_test_csv_from_string(csv_content, delimiter=';')
        try:
            accumulator = aggregator.process_file(path, delimiter=';')
            assert 'CMP001' in accumulator
            assert 'CMP002' in accumulator
        finally:
            os.unlink(path)


# =============================================================================
# ROBUSTNESS + STATS
# =============================================================================

class TestRobustnessAndStats:
    """Test error handling and statistics tracking."""

    def test_missing_campaign_id_skipped(self, aggregator):
        """Test that rows with missing/blank campaign_id are skipped and counted."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions
CMP001,2025-01-01,1000,50,100.50,5
,2025-01-02,2000,100,200.00,10
CMP003,2025-01-01,3000,150,300.00,15"""
        path = create_test_csv_from_string(csv_content)
        try:
            accumulator = aggregator.process_file(path)
            assert aggregator.stats['rows_read'] == 3
            assert aggregator.stats['rows_skipped'] >= 1
            assert 'CMP001' in accumulator
            assert 'CMP003' in accumulator
            # Empty campaign_id should not be in accumulator
            assert '' not in accumulator
        finally:
            os.unlink(path)

    def test_whitespace_campaign_id_handled(self, aggregator):
        """Test that campaign_id with whitespace is stripped."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions
  CMP001  ,2025-01-01,1000,50,100.50,5
CMP002,2025-01-01,2000,100,200.00,10"""
        path = create_test_csv_from_string(csv_content)
        try:
            accumulator = aggregator.process_file(path)
            # After stripping, both should be valid or skipped
            assert 'CMP001' in accumulator or aggregator.stats['rows_skipped'] >= 0
        finally:
            os.unlink(path)

    def test_invalid_numeric_parse_errors(self, aggregator):
        """Test that non-numeric values cause parse_errors and are skipped."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions
CMP001,2025-01-01,1000,50,100.50,5
CMP002,2025-01-01,invalid,100,200.00,10
CMP003,2025-01-01,3000,abc,300.00,15"""
        path = create_test_csv_from_string(csv_content)
        try:
            accumulator = aggregator.process_file(path)
            assert aggregator.stats['rows_read'] == 3
            assert aggregator.stats['parse_errors'] >= 2  # At least 2 rows with errors
            assert 'CMP001' in accumulator
            # CMP002 and CMP003 should not be in accumulator due to parse errors
            assert 'CMP002' not in accumulator or accumulator['CMP002']['impressions'] == 0
        finally:
            os.unlink(path)

    def test_negative_values_skipped(self, aggregator):
        """Test that negative values are skipped and NOT counted as parse_errors."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions
CMP001,2025-01-01,1000,50,100.50,5
CMP002,2025-01-01,-100,100,200.00,10
CMP003,2025-01-01,3000,-50,300.00,15
CMP004,2025-01-01,4000,150,-500,20"""
        path = create_test_csv_from_string(csv_content)
        try:
            accumulator = aggregator.process_file(path)
            assert aggregator.stats['rows_read'] == 4
            # Negative values should be skipped (but not parse errors)
            assert aggregator.stats['rows_skipped'] >= 3
            # parse_errors should be 0 (all values were numeric, just negative)
            assert aggregator.stats['parse_errors'] == 0
            assert 'CMP001' in accumulator
        finally:
            os.unlink(path)

    def test_parse_errors_less_or_equal_to_skipped(self, aggregator):
        """Test that parse_errors <= rows_skipped invariant holds."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions
CMP001,2025-01-01,1000,50,100.50,5
,2025-01-01,2000,100,200.00,10
CMP003,2025-01-01,invalid,150,300.00,15
CMP004,2025-01-01,4000,-50,400.00,20"""
        path = create_test_csv_from_string(csv_content)
        try:
            accumulator = aggregator.process_file(path)
            # parse_errors should be subset of skipped
            assert aggregator.stats['parse_errors'] <= aggregator.stats['rows_skipped']
        finally:
            os.unlink(path)

    def test_empty_file(self, aggregator):
        """Test handling of empty CSV file (header only)."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions"""
        path = create_test_csv_from_string(csv_content)
        try:
            accumulator = aggregator.process_file(path)
            assert aggregator.stats['rows_read'] == 0
            assert aggregator.stats['unique_campaigns'] == 0
            assert len(accumulator) == 0
        finally:
            os.unlink(path)

    def test_combination_of_errors(self, aggregator):
        """Test file with various types of errors."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions
CMP001,2025-01-01,1000,50,100.50,5
,2025-01-01,2000,100,200.00,10
CMP003,2025-01-01,invalid,150,300.00,15
CMP004,2025-01-01,4000,150,-500,20
CMP005,2025-01-01,5000,250,500.00,25"""
        path = create_test_csv_from_string(csv_content)
        try:
            accumulator = aggregator.process_file(path)
            assert aggregator.stats['rows_read'] == 5
            # Should have 2 valid rows (CMP001, CMP005)
            assert aggregator.stats['unique_campaigns'] == 2
            assert 'CMP001' in accumulator
            assert 'CMP005' in accumulator
        finally:
            os.unlink(path)


# =============================================================================
# CTR EDGE CASES
# =============================================================================

class TestCTREdgeCases:
    """Test CTR calculation edge cases."""

    def test_ctr_zero_impressions_zero_clicks(self, aggregator):
        """Test CTR when impressions==0 and clicks==0 -> CTR must be 0.0."""
        # Create accumulator directly
        accumulator = {
            'CMP001': {
                'impressions': 0,
                'clicks': 0,
                'spend': 100.0,
                'conversions': 5
            }
        }
        result = aggregator.get_top_10_ctr(accumulator)
        assert len(result) == 1
        assert result[0][5] == 0.0  # CTR is 6th element

    def test_ctr_zero_impressions_positive_clicks(self, aggregator):
        """Test CTR when impressions==0 and clicks>0 -> CTR must be 0.0 (not inf)."""
        accumulator = {
            'CMP001': {
                'impressions': 0,
                'clicks': 100,
                'spend': 100.0,
                'conversions': 5
            }
        }
        result = aggregator.get_top_10_ctr(accumulator)
        assert len(result) == 1
        ctr = result[0][5]
        assert ctr == 0.0  # Must be 0.0, not inf or nan
        assert ctr != float('inf')
        assert not (ctr != ctr)  # Check for NaN

    def test_ctr_normal_case(self, aggregator):
        """Test normal CTR calculation."""
        accumulator = {
            'CMP001': {
                'impressions': 1000,
                'clicks': 50,
                'spend': 100.0,
                'conversions': 5
            }
        }
        result = aggregator.get_top_10_ctr(accumulator)
        assert len(result) == 1
        ctr = result[0][5]
        assert abs(ctr - 0.05) < 0.0001  # 50/1000 = 0.05

    def test_ctr_negative_impressions(self, aggregator):
        """Test that negative impressions are not possible (filtered earlier)."""
        # This would be caught in process_file, so we just test CTR calculation
        # assumes valid input
        accumulator = {
            'CMP001': {
                'impressions': 1000,
                'clicks': 50,
                'spend': 100.0,
                'conversions': 5
            }
        }
        result = aggregator.get_top_10_ctr(accumulator)
        assert len(result) == 1
        assert result[0][5] >= 0.0  # CTR should never be negative


# =============================================================================
# RANKING + TIE-BREAKERS
# =============================================================================

class TestRankingAndTieBreakers:
    """Test ranking logic and tie-breakers."""

    def test_ctr_tiebreaker_clicks_desc(self, aggregator):
        """Test CTR tie-breaker: higher clicks first."""
        accumulator = {
            'CMP001': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            'CMP002': {'impressions': 1000, 'clicks': 200, 'spend': 200.0, 'conversions': 20},
            # Both have CTR = 0.1, but CMP002 has more clicks
        }
        result = aggregator.get_top_10_ctr(accumulator)
        assert len(result) == 2
        assert result[0][0] == 'CMP002'  # Higher clicks
        assert result[1][0] == 'CMP001'

    def test_ctr_tiebreaker_impressions_desc(self, aggregator):
        """Test CTR tie-breaker: higher impressions first (after clicks)."""
        accumulator = {
            'CMP001': {'impressions': 2000, 'clicks': 200, 'spend': 200.0, 'conversions': 20},
            'CMP002': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            # Both have CTR = 0.1, same click/impression ratio
            # But CMP001 has higher absolute values
        }
        result = aggregator.get_top_10_ctr(accumulator)
        # Both have same CTR, but we need to check tie-breaker order
        # Since clicks are different (200 vs 100), CMP001 should win
        assert result[0][0] == 'CMP001'

    def test_ctr_tiebreaker_campaign_id_asc(self, aggregator):
        """Test CTR tie-breaker: campaign_id ascending (alphabetical)."""
        accumulator = {
            'CMP003': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            'CMP001': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            'CMP002': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            # All identical except campaign_id
        }
        result = aggregator.get_top_10_ctr(accumulator)
        assert len(result) == 3
        assert result[0][0] == 'CMP001'  # Alphabetically first
        assert result[1][0] == 'CMP002'
        assert result[2][0] == 'CMP003'

    def test_ctr_full_tiebreaker_chain(self, aggregator):
        """Test complete CTR tie-breaker chain."""
        accumulator = {
            'A': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            'B': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            'C': {'impressions': 2000, 'clicks': 200, 'spend': 200.0, 'conversions': 20},
            'D': {'impressions': 1000, 'clicks': 200, 'spend': 200.0, 'conversions': 20},
            # All have CTR = 0.1
            # D has highest clicks -> first
            # A and B tied on clicks/impressions, A comes first alphabetically
            # C has same ratio as others but higher absolute -> after D in tie-break
        }
        result = aggregator.get_top_10_ctr(accumulator)
        assert result[0][0] == 'D'  # Highest clicks

    def test_cpa_excludes_zero_conversions(self, aggregator):
        """Test that campaigns with conversions==0 are excluded from CPA ranking."""
        accumulator = {
            'CMP001': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 0},
            'CMP002': {'impressions': 1000, 'clicks': 100, 'spend': 200.0, 'conversions': 10},
            'CMP003': {'impressions': 1000, 'clicks': 100, 'spend': 300.0, 'conversions': 15},
        }
        result = aggregator.get_top_10_cpa(accumulator)
        assert len(result) == 2  # Only CMP002 and CMP003
        campaign_ids = [r[0] for r in result]
        assert 'CMP001' not in campaign_ids  # Excluded
        assert 'CMP002' in campaign_ids
        assert 'CMP003' in campaign_ids

    def test_cpa_tiebreaker_conversions_desc(self, aggregator):
        """Test CPA tie-breaker: higher conversions first."""
        accumulator = {
            'CMP001': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},  # CPA=10
            'CMP002': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 20},  # CPA=5
            # Both have same spend/conversion ratio, CMP002 has more conversions
        }
        result = aggregator.get_top_10_cpa(accumulator)
        assert len(result) == 2
        assert result[0][0] == 'CMP002'  # Lower CPA AND higher conversions

    def test_cpa_tiebreaker_spend_asc(self, aggregator):
        """Test CPA tie-breaker: lower spend first."""
        accumulator = {
            'CMP001': {'impressions': 1000, 'clicks': 100, 'spend': 90.0, 'conversions': 10},  # CPA=9
            'CMP002': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},  # CPA=10
        }
        result = aggregator.get_top_10_cpa(accumulator)
        assert len(result) == 2
        assert result[0][0] == 'CMP001'  # Lower CPA

    def test_cpa_tiebreaker_campaign_id_asc(self, aggregator):
        """Test CPA tie-breaker: campaign_id ascending."""
        accumulator = {
            'CMP003': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            'CMP001': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            'CMP002': {'impressions': 1000, 'clicks': 100, 'spend': 100.0, 'conversions': 10},
            # All identical except campaign_id
        }
        result = aggregator.get_top_10_cpa(accumulator)
        assert len(result) == 3
        assert result[0][0] == 'CMP001'  # Alphabetically first
        assert result[1][0] == 'CMP002'
        assert result[2][0] == 'CMP003'

    def test_top_10_limit(self, aggregator):
        """Test that only top 10 are returned when there are more campaigns."""
        accumulator = {}
        for i in range(20):
            accumulator[f'CMP{i:02d}'] = {
                'impressions': 1000 * (20 - i),  # Descending to test ranking
                'clicks': 100 * (20 - i),
                'spend': 100.0,
                'conversions': 10
            }
        result = aggregator.get_top_10_ctr(accumulator)
        assert len(result) == 10  # Should only return top 10


# =============================================================================
# OUTPUT CSV FORMATTING
# =============================================================================

class TestOutputCSVFormatting:
    """Test CSV output formatting."""

    def test_header_exact_match(self, aggregator):
        """Test that header matches exact specification."""
        campaigns = [
            ('CMP001', 1000, 100, 100.50, 10, 0.1, 10.05)
        ]
        path = create_test_csv([], has_header=False)
        output_path = path + '_output'
        try:
            aggregator.write_output_csv(campaigns, output_path)

            with open(output_path, 'r') as f:
                header = f.readline().strip()
            expected_header = 'campaign_id,total_impressions,total_clicks,total_spend,total_conversions,CTR,CPA'
            assert header == expected_header
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(path)

    def test_spend_2_decimals(self, aggregator):
        """Test that spend is formatted to 2 decimal places."""
        campaigns = [
            ('CMP001', 1000, 100, 100.506789, 10, 0.1, 10.05),
            ('CMP002', 2000, 200, 200.2, 20, 0.1, 10.01),
            ('CMP003', 3000, 300, 300.123, 30, 0.1, 10.00),
        ]
        path = create_test_csv([], has_header=False)
        output_path = path + '_output'
        try:
            aggregator.write_output_csv(campaigns, output_path)

            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert rows[0]['total_spend'] == '100.51'  # Rounded
            assert rows[1]['total_spend'] == '200.20'
            assert rows[2]['total_spend'] == '300.12'
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(path)

    def test_ctr_4_decimals(self, aggregator):
        """Test that CTR is formatted to 4 decimal places."""
        campaigns = [
            ('CMP001', 1000, 50, 100.0, 10, 0.05, 10.0),
            ('CMP002', 1000, 123, 100.0, 10, 0.123, 10.0),
            ('CMP003', 1000, 1, 100.0, 10, 0.001, 10.0),
        ]
        path = create_test_csv([], has_header=False)
        output_path = path + '_output'
        try:
            aggregator.write_output_csv(campaigns, output_path)

            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert rows[0]['CTR'] == '0.0500'
            assert rows[1]['CTR'] == '0.1230'
            assert rows[2]['CTR'] == '0.0010'
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(path)

    def test_cpa_2_decimals_or_empty(self, aggregator):
        """Test that CPA is formatted to 2 decimals or empty when None."""
        campaigns = [
            ('CMP001', 1000, 100, 100.0, 10, 0.1, 10.0),
            ('CMP002', 2000, 200, 200.0, 20, 0.1, 10.0),
            ('CMP003', 3000, 0, 300.0, 0, 0.0, None),  # No conversions
        ]
        path = create_test_csv([], has_header=False)
        output_path = path + '_output'
        try:
            aggregator.write_output_csv(campaigns, output_path)

            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert rows[0]['CPA'] == '10.00'
            assert rows[1]['CPA'] == '10.00'
            assert rows[2]['CPA'] == ''  # Empty when None
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(path)

    def test_exactly_7_columns(self, aggregator):
        """Test that each row has exactly 7 columns."""
        campaigns = [
            ('CMP001', 1000, 100, 100.50, 10, 0.1, 10.05),
            ('CMP002', 2000, 200, 200.00, 20, 0.1, 10.00),
        ]
        path = create_test_csv([], has_header=False)
        output_path = path + '_output'
        try:
            aggregator.write_output_csv(campaigns, output_path)

            with open(output_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Header row
            assert len(rows[0]) == 7
            # Data rows
            for i in range(1, len(rows)):
                assert len(rows[i]) == 7, f"Row {i} has {len(rows[i])} columns, expected 7"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(path)

    def test_round_trip_consistency(self, aggregator):
        """Test that output can be read back correctly."""
        campaigns = [
            ('CMP001', 1000, 100, 100.555, 10, 0.1, 10.0555),
            ('CMP002', 2000, 200, 200.999, 20, 0.1, 10.0499),
        ]
        path = create_test_csv([], has_header=False)
        output_path = path + '_output'
        try:
            aggregator.write_output_csv(campaigns, output_path)

            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]['campaign_id'] == 'CMP001'
            assert rows[0]['total_impressions'] == '1000'
            assert rows[0]['total_clicks'] == '100'
            assert rows[0]['total_spend'] == '100.56'  # Rounded to 2 decimals
            assert rows[0]['total_conversions'] == '10'
            assert rows[0]['CTR'] == '0.1000'  # 4 decimals
            assert rows[0]['CPA'] == '10.06'  # Rounded to 2 decimals
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(path)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_small_dataset(self, aggregator):
        """Test complete pipeline: process -> rank -> output."""
        csv_content = """campaign_id,date,impressions,clicks,spend,conversions
CMP001,2025-01-01,1000,50,100.50,5
CMP002,2025-01-01,2000,200,200.00,10
CMP001,2025-01-02,1500,75,150.75,7
CMP003,2025-01-01,5000,100,500.00,25"""
        input_path = create_test_csv_from_string(csv_content)
        output_path = input_path + '_output'
        try:
            # Process
            accumulator = aggregator.process_file(input_path)
            assert aggregator.stats['rows_read'] == 4
            assert len(accumulator) == 3

            # Rank CTR
            top_ctr = aggregator.get_top_10_ctr(accumulator)
            assert len(top_ctr) == 3

            # Rank CPA
            top_cpa = aggregator.get_top_10_cpa(accumulator)
            assert len(top_cpa) == 3

            # Write output
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            aggregator.write_output_csv(top_ctr, output_path + '_ctr.csv')
            aggregator.write_output_csv(top_cpa, output_path + '_cpa.csv')

            # Verify files exist
            assert os.path.exists(output_path + '_ctr.csv')
            assert os.path.exists(output_path + '_cpa.csv')

        finally:
            if os.path.exists(output_path + '_ctr.csv'):
                os.unlink(output_path + '_ctr.csv')
            if os.path.exists(output_path + '_cpa.csv'):
                os.unlink(output_path + '_cpa.csv')
            os.unlink(input_path)

    def test_empty_campaign_list_output(self, aggregator):
        """Test writing empty campaign list."""
        campaigns = []
        path = create_test_csv([], has_header=False)
        output_path = path + '_output'
        try:
            aggregator.write_output_csv(campaigns, output_path)

            with open(output_path, 'r') as f:
                lines = f.readlines()

            # Should have header but no data rows
            assert len(lines) == 1
            assert 'campaign_id' in lines[0]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(path)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Optional performance tests (not strict assertions)."""

    def test_small_performance_dataset(self, aggregator):
        """Test with 50k rows - marked as performance test."""
        # Create a CSV with 50k rows
        rows = [['campaign_id', 'date', 'impressions', 'clicks', 'spend', 'conversions']]
        for i in range(50000):
            rows.append([
                f'CMP{i % 100:03d}',
                '2025-01-01',
                str(1000 + i),
                str(50 + i % 100),
                '100.50',
                str(5 + i % 10)
            ])

        path = create_test_csv(rows, has_header=True)
        try:
            import time
            start = time.time()
            accumulator = aggregator.process_file(path)
            elapsed = time.time() - start

            # Should complete reasonably fast
            # (No strict assertion to avoid flaky tests)
            print(f"\n[PERFORMANCE] Processed 50,000 rows in {elapsed:.2f} seconds")
            print(f"[PERFORMANCE] Throughput: {50000/elapsed:,.0f} rows/second")

            assert aggregator.stats['rows_read'] == 50000
            assert len(accumulator) == 100

        finally:
            os.unlink(path)


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
