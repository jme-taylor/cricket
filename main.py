from cricket.extraction.json_parsing import JsonDataProcessor   
from cricket.extraction.download import download_and_extract_zipped_cricket_data

if __name__ == "__main__":
    download_and_extract_zipped_cricket_data()
    json_data_processor = JsonDataProcessor()
    json_data_processor.parse_all_matches()
