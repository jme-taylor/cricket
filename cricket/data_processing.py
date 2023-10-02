import json
import os
from pathlib import Path

from cricket.constants import DATA_FOLDER


def load_json_file(file_name: str) -> dict:
    with open(DATA_FOLDER.joinpath(file_name)) as f:
        return json.load(f)


def load_test_matches() -> list:
    test_matches = []
    for match_file in os.listdir(DATA_FOLDER):
        match = load_json_file(match_file)
        if match["info"]["match_type"] == "Test":
            test_matches.append(match)
    return test_matches
