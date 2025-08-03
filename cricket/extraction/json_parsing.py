import polars as pl
from pathlib import Path

from cricket.constants import DATA_FOLDER, INPUT_DATA_FOLDER
from cricket.extraction.match_processing import Match
from cricket.logging_config import logger


class JsonDataProcessor:
    """
    Parse all matches in a folder into parquet files using Pydantic validation.

    Takes a folder of JSON files containing data about cricket matches and parses them 
    using validated Pydantic models for better type safety and data validation.

    Attributes
    ----------
    data_folder : Path
        The folder containing the JSON files to parse. By default, INPUT_DATA_FOLDER
    output_folder : Path
        The folder to output the parsed files to. By default, DATA_FOLDER
    ball_by_ball_filename : str
        The filename of the output parquet file containing the parsed ball-by-ball data
    match_metadata_filename : str
        The filename of the output parquet file containing the parsed match metadata
    """

    def __init__(
        self,
        data_folder: Path = INPUT_DATA_FOLDER,
        output_folder: Path = DATA_FOLDER,
        ball_by_ball_filename: str = "ball_level_data.parquet",
        match_metadata_filename: str = "match_metadata.parquet",
    ):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.ball_by_ball_filename = ball_by_ball_filename
        self.match_metadata_filename = match_metadata_filename

    def parse_all_matches(self):
        """
        Parse all matches in the data folder, saving ball by ball data, and match metadata.
        Uses Pydantic models for validation and type safety.
        """
        matches = []
        match_metadata = []
        all_matches = list(self.data_folder.glob("*.json"))
        logger.info(f"Found {len(all_matches)} matches")
        count_parsed = 0
        
        for match_file in all_matches:
            match = Match(
                match_filepath=match_file,
                match_id=match_file.stem
            )
            
            matches.extend(match.parse_match_data())
            match_metadata.append(match.get_match_metadata())
            count_parsed += 1
            
            if count_parsed % 100 == 0:
                logger.info(f"Parsed {count_parsed} matches")
                    

        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True)

        if len(matches) > 0:
            match_dataframe = pl.from_dicts(matches)
            match_dataframe.write_parquet(
                self.output_folder.joinpath(self.ball_by_ball_filename)
            )

            match_metadata_dataframe = pl.from_dicts(match_metadata)
            match_metadata_dataframe.write_parquet(
                self.output_folder.joinpath(self.match_metadata_filename)
            )
            
            logger.info(f"Successfully processed {len(matches)} balls from {count_parsed} matches")
        else:
            logger.warning("No matches were successfully processed")
