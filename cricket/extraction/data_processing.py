import json
from pathlib import Path
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_json(filepath: Path) -> dict:
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


def load_json_as_model(filepath: Path, model_class: Type[T]) -> T:
    """
    Load a JSON file and parse it into a Pydantic model with validation.

    Parameters
    ----------
    filepath : Path
        The filepath of the JSON file to load.
    model_class : Type[T]
        The Pydantic model class to parse the JSON into.

    Returns
    -------
    T
        An instance of the specified Pydantic model containing the validated JSON data.
    """
    with open(filepath, "r") as f:
        json_data = json.load(f)
    return model_class(**json_data)
