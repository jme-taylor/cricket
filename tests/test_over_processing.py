import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cricket.over_processing import Over

DATA_DIR = Path(__file__).parent.joinpath("test_over_processing")


@pytest.fixture
def scoring_over():
    with open(DATA_DIR.joinpath("scoring_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def wicket_over():
    with open(DATA_DIR.joinpath("wicket_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def maiden_over():
    with open(DATA_DIR.joinpath("maiden_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def extra_over():
    with open(DATA_DIR.joinpath("extra_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def short_over():
    with open(DATA_DIR.joinpath("short_over_input.json")) as f:
        return Over(json.load(f))


def test_parse_over_data(
    scoring_over, wicket_over, maiden_over, extra_over, short_over
):
    expected_scoring_over = pl.read_parquet(
        DATA_DIR.joinpath("scoring_over_output.parquet")
    )
    scoring_over_output = scoring_over.parse_over_data()
    assert_frame_equal(scoring_over_output, expected_scoring_over)
    expected_wicket_over = pl.read_parquet(
        DATA_DIR.joinpath("wicket_over_output.parquet")
    )
    wicket_over_output = wicket_over.parse_over_data()
    assert_frame_equal(wicket_over_output, expected_wicket_over)
    expected_maiden_over = pl.read_parquet(
        DATA_DIR.joinpath("maiden_over_output.parquet")
    )
    maiden_over_output = maiden_over.parse_over_data()
    assert_frame_equal(maiden_over_output, expected_maiden_over)
    expected_extra_over = pl.read_parquet(
        DATA_DIR.joinpath("extra_over_output.parquet")
    )
    extra_over_output = extra_over.parse_over_data()
    assert_frame_equal(extra_over_output, expected_extra_over)
    expected_short_over = pl.read_parquet(
        DATA_DIR.joinpath("short_over_output.parquet")
    )
    short_over_output = short_over.parse_over_data()
    assert_frame_equal(short_over_output, expected_short_over)
