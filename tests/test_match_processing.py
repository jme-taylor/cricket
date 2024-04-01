from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cricket.match_processing import Match

DATA_DIR = Path(__file__).parent.joinpath("test_match_processing")


@pytest.fixture
def test_test_match():
    return Match(DATA_DIR.joinpath("test_match_input.json"), match_id=1)


@pytest.fixture
def test_odi_match():
    return Match(DATA_DIR.joinpath("odi_match_input.json"), match_id=2)


@pytest.fixture
def test_t20_match():
    return Match(DATA_DIR.joinpath("t20_match_input.json"), match_id=3)


@pytest.fixture
def test_hundred_match():
    return Match(DATA_DIR.joinpath("hundred_match_input.json"), match_id=4)


@pytest.fixture
def test_forfeitted_match():
    return Match(
        DATA_DIR.joinpath("innings_forfeit_match_input.json"), match_id=5
    )


def test_lookup_player(test_test_match):
    assert test_test_match.lookup_player("AC Gilchrist") == "2b6e6dec"
    assert test_test_match.lookup_player("David Beckham") is None


def test_get_match_metadata(
    test_test_match,
    test_odi_match,
    test_t20_match,
    test_hundred_match,
    test_forfeitted_match,
):
    test_match_metadata = pl.read_parquet(
        DATA_DIR.joinpath("test_match_metadata.parquet")
    )
    assert_frame_equal(
        test_test_match.get_match_metadata(), test_match_metadata
    )
    odi_match_metadata = pl.read_parquet(
        DATA_DIR.joinpath("odi_match_metadata.parquet")
    )
    assert_frame_equal(test_odi_match.get_match_metadata(), odi_match_metadata)
    t20_match_metadata = pl.read_parquet(
        DATA_DIR.joinpath("t20_match_metadata.parquet")
    )
    assert_frame_equal(test_t20_match.get_match_metadata(), t20_match_metadata)
    hundred_match_metadata = pl.read_parquet(
        DATA_DIR.joinpath("hundred_match_metadata.parquet")
    )
    assert_frame_equal(
        test_hundred_match.get_match_metadata(), hundred_match_metadata
    )
    forfeit_match_metadata = pl.read_parquet(
        DATA_DIR.joinpath("innings_forfeit_match_metadata.parquet")
    )
    assert_frame_equal(
        test_forfeitted_match.get_match_metadata(), forfeit_match_metadata
    )


def test_parse_match_data(
    test_test_match,
    test_odi_match,
    test_t20_match,
    test_hundred_match,
    test_forfeitted_match,
):
    test_match_data = pl.read_parquet(
        DATA_DIR.joinpath("test_match_output.parquet")
    )
    assert_frame_equal(test_test_match.parse_match_data(), test_match_data)
    odi_match_data = pl.read_parquet(
        DATA_DIR.joinpath("odi_match_output.parquet")
    )
    assert_frame_equal(test_odi_match.parse_match_data(), odi_match_data)
    t20_match_data = pl.read_parquet(
        DATA_DIR.joinpath("t20_match_output.parquet")
    )
    assert_frame_equal(test_t20_match.parse_match_data(), t20_match_data)
    hundred_match_data = pl.read_parquet(
        DATA_DIR.joinpath("hundred_match_output.parquet")
    )
    assert_frame_equal(
        test_hundred_match.parse_match_data(), hundred_match_data
    )
    forfeit_match_data = pl.read_parquet(
        DATA_DIR.joinpath("innings_forfeit_match_output.parquet")
    )
    assert_frame_equal(
        test_forfeitted_match.parse_match_data(), forfeit_match_data
    )
