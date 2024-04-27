from pathlib import Path

from cricket.json_parsing import JsonDataProcessor

TEST_DATA_FOLDER = Path(__file__).parent.joinpath("test_json_parsing")
TEST_DATA_INPUT_FOLDER = TEST_DATA_FOLDER.joinpath("test_input")


def test_parse_all_matches():
    json_data_processor = JsonDataProcessor(
        data_folder=TEST_DATA_INPUT_FOLDER, output_folder=TEST_DATA_FOLDER
    )

    json_data_processor.parse_all_matches()

    ball_by_ball_filepath = (
        json_data_processor.output_folder.joinpath("ball_level_data.parquet")
    )
    match_metadata_filepath = (
        json_data_processor.output_folder.joinpath("match_metadata.parquet")
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
        json_data_processor.data_folder.joinpath("ball_level_data.parquet")
    )
    match_metadata_filepath = (
        json_data_processor.data_folder.joinpath("match_metadata.parquet")
    )

    assert not ball_by_ball_filepath.exists()
    assert not match_metadata_filepath.exists()
