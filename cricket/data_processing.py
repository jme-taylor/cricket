import json
from pathlib import Path
from typing import Dict


def load_json(filepath: Path) -> Dict:
    """
    Load a json from a filepath and return as a dictionary.

    Parameters
    ----------
    filepath : Path
        The filepath of the JSON file to load.

    Returns
    -------
    dict
        A dictionary containing the JSON data.
    """
    with open(filepath, "r") as f:
        return json.load(f)
