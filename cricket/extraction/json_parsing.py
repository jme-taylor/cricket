import polars as pl

from cricket.constants import DATA_FOLDER, INPUT_DATA_FOLDER
from cricket.extraction.match_processing import Match
from cricket.logging_config import logger


class JsonDataProcessor:
    """
    Parse all matches in a folder into a single JSON file, as well as a JSON file containing the match metadata.

    Takes a folder of JSON files containing data about cricket matches and parses them into a single JSON file.

    Attributes
    ----------
    data_folder : Path
        The folder containing the JSON files to parse. By default, INPUT_DATA_FOLDER (defined in constants.py)
    output_folder : Path
        The folder to output the parsed JSON file to. By default, DATA_FOLDER (defined in constants.py)
    ball_by_ball_filename : str
        The filename of the output JSON file containing the parsed ball-by-ball data. By default, "all_ball_by_ball.jsonl"
    match_metadata_filename : str
        The filename of the output JSON file containing the parsed match metadata. By default, "all_match_metadata.jsonl"
    """

    def __init__(
        self,
        data_folder=INPUT_DATA_FOLDER,
        output_folder=DATA_FOLDER,
        ball_by_ball_filename="ball_level_data.parquet",
        match_metadata_filename="match_metadata.parquet",
    ):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.ball_by_ball_filename = ball_by_ball_filename
        self.match_metadata_filename = match_metadata_filename

    def parse_all_matches(self):
        """
        Parse all matches in the data folder, saving ball by ball data, and match metadata.
        """
        matches = []
        match_metadata = []
        all_matches = list(self.data_folder.glob("*.json"))
        logger.info(f"Found {len(all_matches)} matches")
        count_parsed = 0
        for match_file in all_matches:
            match = Match(match_file, match_file.stem)
            matches.extend(match.parse_match_data())
            match_metadata.append(match.get_match_metadata())
            count_parsed += 1
            if count_parsed % 100 == 0:
                logger.info(f"Parsed {count_parsed} matches")

        if self.output_folder.exists() is False:
            self.output_folder.mkdir(parents=True)

        if len(matches) != 0:
            match_dataframe = pl.from_dicts(matches)
            match_dataframe.write_parquet(
                self.output_folder.joinpath(self.ball_by_ball_filename)
            )

            match_metadata_dataframe = pl.from_dicts(match_metadata)
            match_metadata_dataframe.write_parquet(
                self.output_folder.joinpath(self.match_metadata_filename)
            )
