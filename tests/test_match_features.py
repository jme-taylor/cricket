"""
Tests for match state features using TDD approach with real match data.
Uses match 1072317 (England vs Australia T20) as test fixture.
"""

import pytest
import polars as pl
from pathlib import Path

# Import the functions we'll be implementing
from cricket.transformation.match import (
    get_current_score,
    get_wickets_fallen,
    get_overs_remaining,
)


# Test fixtures using real match data
@pytest.fixture
def match_1072317_first_over():
    """Load first over of match 1072317 for basic tests"""
    test_dir = Path("tests/test_match_features")
    return pl.read_parquet(test_dir / "match_1072317_first_over.parquet")


@pytest.fixture
def match_1072317_both_innings():
    """Load both innings (10 overs each) for innings transition tests"""
    test_dir = Path("tests/test_match_features")
    return pl.read_parquet(test_dir / "match_1072317_both_innings_10_overs.parquet")


@pytest.fixture
def match_1072317_full():
    """Load complete match for integration tests"""
    test_dir = Path("tests/test_match_features")
    return pl.read_parquet(test_dir / "match_1072317_full_match.parquet")


@pytest.fixture
def match_1072317_first_50():
    """Load first 50 balls for medium-sized tests"""
    test_dir = Path("tests/test_match_features")
    return pl.read_parquet(test_dir / "match_1072317_first_50_balls.parquet")


class TestGetCurrentScore:
    """Test cases for get_current_score function"""

    def test_basic_score_accumulation(self, match_1072317_first_50):
        """Test that scores accumulate correctly ball by ball using real match data"""
        input_df = match_1072317_first_50

        result = get_current_score(input_df)

        # Test that current_score column exists and is calculated correctly
        assert "current_score" in result.columns

        # Score should start at 0
        assert result["current_score"][0] == 0

        # Each subsequent score should be cumulative sum of previous runs
        for i in range(1, len(result)):
            expected_score = result["runs"][:i].sum()
            assert result["current_score"][i] == expected_score

        # Score should never decrease (monotonic)
        scores = result["current_score"].to_list()
        assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

    def test_score_resets_between_innings(self, match_1072317_both_innings):
        """Test that scores reset for each new innings using real match data"""
        input_df = match_1072317_both_innings

        result = get_current_score(input_df)

        # Check that we have both innings
        innings = result["innings_number"].unique().sort()
        assert len(innings) >= 2, "Test data should contain both innings"

        # Check that each innings starts with score 0
        for innings_num in innings:
            innings_data = result.filter(pl.col("innings_number") == innings_num)
            first_ball_score = innings_data["current_score"][0]
            assert first_ball_score == 0, (
                f"Innings {innings_num} should start with score 0"
            )

        # Check that scores within each innings are monotonic
        for innings_num in innings:
            innings_data = result.filter(pl.col("innings_number") == innings_num)
            scores = innings_data["current_score"].to_list()
            assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)), (
                f"Scores within innings {innings_num} should be monotonic"
            )

    def test_empty_dataframe(self):
        """Test handling of empty input"""
        input_df = pl.DataFrame(
            {
                "match_id": [],
                "innings_number": [],
                "runs": [],
                "delivery": [],
                "team": [],
                "batter": [],
                "bowler": [],
                "over_num": [],
                "ball_num": [],
            }
        ).cast(
            {
                "match_id": pl.Utf8,
                "innings_number": pl.Int64,
                "runs": pl.Int64,
                "delivery": pl.Float64,
                "team": pl.Utf8,
                "batter": pl.Utf8,
                "bowler": pl.Utf8,
                "over_num": pl.Int64,
                "ball_num": pl.Int64,
            }
        )

        result = get_current_score(input_df)

        assert "current_score" in result.columns
        assert result.shape[0] == 0


class TestGetWicketsFallen:
    """Test cases for get_wickets_fallen function"""

    def test_basic_wicket_tracking(self, match_1072317_first_50):
        """Test wickets accumulate correctly"""
        input_df = match_1072317_first_50

        result = get_wickets_fallen(input_df)

        # Test that wickets_fallen column exists
        assert "wickets_fallen" in result.columns

        # First ball should have 0 wickets fallen
        assert result["wickets_fallen"][0] == 0

        # Wickets should never decrease (monotonic)
        wickets = result["wickets_fallen"].to_list()
        assert all(wickets[i] <= wickets[i + 1] for i in range(len(wickets) - 1))

        # Check that wickets only increase when there's a dismissal
        for i in range(1, len(result)):
            wicket_increase = (
                result["wickets_fallen"][i] - result["wickets_fallen"][i - 1]
            )
            has_wicket = (
                result["player_out_1"][i - 1] != ""
                or result["player_out_2"][i - 1] != ""
            )

            if has_wicket:
                assert wicket_increase >= 1, (
                    f"Wickets should increase when there's a dismissal at index {i - 1}"
                )
            else:
                assert wicket_increase == 0, (
                    f"Wickets should not increase without dismissal at index {i - 1}"
                )

    def test_wickets_reset_between_innings(self, match_1072317_both_innings):
        """Test wickets reset for new innings"""
        input_df = match_1072317_both_innings

        result = get_wickets_fallen(input_df)

        # Check that we have both innings
        innings = result["innings_number"].unique().sort()
        assert len(innings) >= 2

        # Check that each innings starts with 0 wickets
        for innings_num in innings:
            innings_data = result.filter(pl.col("innings_number") == innings_num)
            first_ball_wickets = innings_data["wickets_fallen"][0]
            assert first_ball_wickets == 0, (
                f"Innings {innings_num} should start with 0 wickets fallen"
            )

    def test_maximum_ten_wickets(self, match_1072317_full):
        """Test that wickets cap at 10 (all out)"""
        input_df = match_1072317_full

        result = get_wickets_fallen(input_df)

        # Should never exceed 10 wickets in any innings
        max_wickets_per_innings = result.group_by(["match_id", "innings_number"]).agg(
            pl.col("wickets_fallen").max().alias("max_wickets")
        )

        assert all(max_wickets_per_innings["max_wickets"] <= 10), (
            "No innings should have more than 10 wickets fallen"
        )


class TestGetOversRemaining:
    """Test cases for get_overs_remaining function"""

    def test_t20_overs_remaining(self, match_1072317_first_50):
        """Test overs remaining in T20 (20 overs total)"""
        input_df = match_1072317_first_50

        result = get_overs_remaining(input_df)

        # Test that overs_remaining column exists
        assert "overs_remaining" in result.columns

        # First ball should have close to 20 overs remaining (T20 match)
        # After 1 legal delivery, should be 20 - 1/6 = 19.833...
        assert abs(result["overs_remaining"][0] - (20.0 - 1 / 6)) < 0.001

        # Overs remaining should generally decrease (allowing for extras)
        overs_remaining = result["overs_remaining"].to_list()

        # Check that overs remaining behaves correctly
        for i in range(1, len(result)):
            prev_overs = overs_remaining[i - 1]
            curr_overs = overs_remaining[i]

            # Check the CURRENT ball to see if it's an extra
            current_ball_is_extra = result["wides"][i] > 0 or result["noballs"][i] > 0

            if current_ball_is_extra:
                # For extras, overs should remain the same
                assert prev_overs == curr_overs, (
                    f"Overs should not decrease for extras at index {i}"
                )
            else:
                # For legal deliveries, overs should decrease by 1/6
                expected_decrease = 1 / 6
                actual_decrease = prev_overs - curr_overs
                assert abs(actual_decrease - expected_decrease) < 0.001, (
                    f"Legal delivery should decrease overs by 1/6 at index {i}. Expected decrease: {expected_decrease:.3f}, Actual: {actual_decrease:.3f}"
                )

    def test_overs_with_extras(self, match_1072317_first_50):
        """Test that wides/no-balls don't consume legal deliveries"""
        input_df = match_1072317_first_50

        result = get_overs_remaining(input_df)

        # Find balls with extras
        extras_mask = (result["wides"] > 0) | (result["noballs"] > 0)

        if extras_mask.sum() > 0:  # Only test if there are extras in the sample
            # Test the specific pattern we know exists
            # From our debugging: ball index 2 is a wide at delivery 0.3
            wide_indices = [i for i in range(len(result)) if result["wides"][i] > 0]

            if len(wide_indices) > 0:
                wide_idx = wide_indices[0]  # First wide ball

                # Wide ball should have same overs remaining as the ball before it
                if wide_idx > 0:
                    before_wide = result["overs_remaining"][wide_idx - 1]
                    wide_ball = result["overs_remaining"][wide_idx]

                    # Wide ball should have same overs remaining as ball before it
                    assert before_wide == wide_ball, (
                        f"Wide ball should have same overs as previous ball. Before: {before_wide}, Wide: {wide_ball}"
                    )

    def test_second_innings_with_target(self, match_1072317_both_innings):
        """Test overs remaining in second innings (chasing)"""
        input_df = match_1072317_both_innings

        # Filter to second innings only
        second_innings = input_df.filter(pl.col("innings_number") == 2)

        if len(second_innings) > 0:  # Only test if second innings exists
            result = get_overs_remaining(second_innings)

            # Second innings should also start with appropriate overs remaining
            # For T20, should be close to 20 (or target_overs if D/L applied)
            first_ball_overs = result["overs_remaining"][0]

            # Should be between 0 and 20 overs
            assert 0 <= first_ball_overs <= 20, (
                f"Second innings should start with valid overs remaining, got {first_ball_overs}"
            )


class TestIntegration:
    """Integration tests using all features together"""

    def test_all_features_together(self, match_1072317_first_50):
        """Test that all features work together without conflicts"""
        input_df = match_1072317_first_50

        # Apply all features
        result = get_current_score(input_df)
        result = get_wickets_fallen(result)
        result = get_overs_remaining(result)

        # Check that all feature columns exist
        assert "current_score" in result.columns
        assert "wickets_fallen" in result.columns
        assert "overs_remaining" in result.columns

        # Check that the DataFrame still has the same number of rows
        assert len(result) == len(input_df)

        # Check that all values are reasonable
        assert all(result["current_score"] >= 0)
        assert all(result["wickets_fallen"] >= 0)
        assert all(result["wickets_fallen"] <= 10)
        assert all(result["overs_remaining"] >= 0)
        assert all(result["overs_remaining"] <= 20)  # T20 match

    def test_full_match_final_ball_england_innings(self, match_1072317_full):
        """Test final ball of England innings shows correct match state"""
        input_df = match_1072317_full

        # Apply all features
        result = get_current_score(input_df)
        result = get_wickets_fallen(result)
        result = get_overs_remaining(result)

        # Filter to England innings (innings 1)
        england_innings = result.filter(pl.col("innings_number") == 1)

        # Get the last ball of England's innings
        final_ball = england_innings.tail(1)

        # Verify the match state for the final ball of England's innings
        assert final_ball["current_score"][0] == 149, (
            f"Final ball should show 149 runs, got {final_ball['current_score'][0]}"
        )
        assert final_ball["wickets_fallen"][0] == 9, (
            f"Final ball should show 9 wickets fallen, got {final_ball['wickets_fallen'][0]}"
        )
        assert final_ball["overs_remaining"][0] == 0, (
            f"Final ball should show 0 overs remaining, got {final_ball['overs_remaining'][0]}"
        )
