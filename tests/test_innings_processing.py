import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cricket.innings_processing import Innings

DATA_DIR = Path(__file__).parent.joinpath("test_innings_processing")


@pytest.fixture
def test_innings():
    with open(DATA_DIR.joinpath("innings_input.json")) as f:
        return Innings(json.load(f), 1)


@pytest.fixture
def test_innings_with_powerplay():
    with open(DATA_DIR.joinpath("innings_with_powerplay_input.json")) as f:
        return Innings(json.load(f), 1)


@pytest.fixture
def test_innings_with_target():
    with open(DATA_DIR.joinpath("innings_with_target_input.json")) as f:
        return Innings(json.load(f), 1)


def test_parse_innings_data(
    test_innings, test_innings_with_powerplay, test_innings_with_target
):
    expected_innings_output = pl.read_parquet(
        DATA_DIR.joinpath("innings_output.parquet")
    )
    innings_output = test_innings.parse_innings_data()
    assert_frame_equal(expected_innings_output, innings_output)
    expected_innings_with_powerplay_output = pl.read_parquet(
        DATA_DIR.joinpath("innings_with_powerplay_output.parquet")
    )
    innings_with_powerplay_output = (
        test_innings_with_powerplay.parse_innings_data()
    )
    assert_frame_equal(
        expected_innings_with_powerplay_output, innings_with_powerplay_output
    )
    expected_innings_with_target_output = pl.read_parquet(
        DATA_DIR.joinpath("innings_with_target_output.parquet")
    )
    innings_with_target_output = test_innings_with_target.parse_innings_data()
    assert_frame_equal(
        expected_innings_with_target_output, innings_with_target_output
    )
