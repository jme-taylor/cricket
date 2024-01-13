import json
from pathlib import Path

import pytest

from cricket.over_processing import Over

TEST_DATA_DIR = Path(__file__).parent.joinpath("test_data")


@pytest.fixture
def scoring_over():
    with open(TEST_DATA_DIR.joinpath("scoring_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def wicket_over():
    with open(TEST_DATA_DIR.joinpath("wicket_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def maiden_over():
    with open(TEST_DATA_DIR.joinpath("maiden_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def extra_over():
    with open(TEST_DATA_DIR.joinpath("extra_over_input.json")) as f:
        return Over(json.load(f))


@pytest.fixture
def short_over():
    with open(TEST_DATA_DIR.joinpath("short_over_input.json")) as f:
        return Over(json.load(f))


def test_parse_over_data(
    scoring_over, wicket_over, maiden_over, extra_over, short_over
):
    with open(TEST_DATA_DIR.joinpath("scoring_over_output.json")) as f:
        assert scoring_over.parse_over_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("wicket_over_output.json")) as f:
        assert wicket_over.parse_over_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("maiden_over_output.json")) as f:
        assert maiden_over.parse_over_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("extra_over_output.json")) as f:
        assert extra_over.parse_over_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("short_over_output.json")) as f:
        assert short_over.parse_over_data() == json.load(f)
