import json
from pathlib import Path

import pytest

from cricket.extraction.data_processing import load_json

TEST_JSON_DATA = {"key": "value"}


@pytest.fixture
def json_file(tmp_path):
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(TEST_JSON_DATA, f)
    return file_path


def test_load_json(json_file):
    result = load_json(json_file)
    assert result == TEST_JSON_DATA


def test_load_json_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_json(Path("nonexistent_file.json"))


def test_load_json_invalid_json(tmp_path):
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, "w") as f:
        f.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
        load_json(invalid_json_file)
