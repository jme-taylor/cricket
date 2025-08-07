#!/usr/bin/env python3
"""
Create test fixture from match 1072317 for use in tests.
This is a one-time script to generate realistic test data.
"""

import polars as pl
import json
from pathlib import Path
from cricket.extraction.match_processing import Match


def create_match_1072317_fixture():
    """
    Create test fixture from match 1072317 for use in tests.
    This is a one-time script to generate realistic test data.
    """
    # Process the specific match
    match_file = Path("input_data/1072317.json")
    match = Match(match_filepath=match_file, match_id="1072317")

    # Get ball-by-ball data
    ball_data = match.parse_match_data()
    ball_df = pl.from_dicts(ball_data)

    # Get match metadata
    metadata = match.get_match_metadata()
    metadata_df = pl.from_dicts([metadata])

    # Join to get complete dataset (like get_all_matches_data does)
    complete_df = ball_df.join(metadata_df, on="match_id")

    # Create test output directory
    test_dir = Path("tests/test_match_features")
    test_dir.mkdir(exist_ok=True)

    # Save different sized samples for different test scenarios

    # 1. First 50 balls for basic tests
    first_50_balls = complete_df.head(50)
    first_50_balls.write_parquet(test_dir / "match_1072317_first_50_balls.parquet")

    # 2. First complete over (6 balls) for simple tests
    first_over = complete_df.filter(pl.col("over_num") == 0)
    first_over.write_parquet(test_dir / "match_1072317_first_over.parquet")

    # 3. Both innings first 10 overs for innings transition tests
    both_innings_sample = complete_df.filter(pl.col("over_num") < 10)
    both_innings_sample.write_parquet(
        test_dir / "match_1072317_both_innings_10_overs.parquet"
    )

    # 4. Full match for integration tests
    complete_df.write_parquet(test_dir / "match_1072317_full_match.parquet")

    # 5. Create summary info for documentation
    match_info = {
        "match_id": "1072317",
        "teams": [metadata["team_1"], metadata["team_2"]],
        "match_type": metadata["match_type"],
        "venue": metadata["venue"],
        "total_balls": len(ball_data),
        "innings_1_balls": len([b for b in ball_data if b["innings_number"] == 1]),
        "innings_2_balls": len([b for b in ball_data if b["innings_number"] == 2]),
        "wickets_innings_1": len(
            [
                b
                for b in ball_data
                if b["innings_number"] == 1 and b["player_out_1"] != ""
            ]
        ),
        "wickets_innings_2": len(
            [
                b
                for b in ball_data
                if b["innings_number"] == 2 and b["player_out_1"] != ""
            ]
        ),
    }

    # Save match info as JSON for reference
    with open(test_dir / "match_1072317_info.json", "w") as f:
        json.dump(match_info, f, indent=2, default=str)

    print(f"Created test fixtures for match {match_info['match_id']}")
    print(f"Match: {match_info['teams'][0]} vs {match_info['teams'][1]}")
    print(f"Format: {match_info['match_type']}")
    print(f"Total balls: {match_info['total_balls']}")
    print(f"Files created in: {test_dir}")

    return match_info


if __name__ == "__main__":
    create_match_1072317_fixture()
