from pathlib import Path

from cricket.constants import PROJECT_ROOT
from cricket.json_parsing import JsonDataProcessor

TEST_DATA_FOLDER = PROJECT_ROOT.joinpath("tests", "test_data")
INPUT_FOLDER = TEST_DATA_FOLDER.joinpath("test_input")


def test_parse_all_matches():
    json_data_processor = JsonDataProcessor(
        data_folder=INPUT_FOLDER, output_folder=TEST_DATA_FOLDER
    )

    json_data_processor.parse_all_matches()

    ball_by_ball_filepath = (
        json_data_processor.output_folder / "all_ball_by_ball.jsonl"
    )
    match_metadata_filepath = (
        json_data_processor.output_folder / "all_match_metadata.jsonl"
    )

    assert ball_by_ball_filepath.exists()
    assert match_metadata_filepath.exists()


def test_parse_all_matches_no_matches():
    # Test when there are no matches to parse
    json_data_processor = JsonDataProcessor(
        data_folder=Path("nonexistent_folder")
    )
    json_data_processor.parse_all_matches()

    ball_by_ball_filepath = (
        json_data_processor.data_folder / "test_ball_by_ball.jsonl"
    )
    match_metadata_filepath = (
        json_data_processor.data_folder / "test_match_metadata.jsonl"
    )

    assert not ball_by_ball_filepath.exists()
    assert not match_metadata_filepath.exists()
