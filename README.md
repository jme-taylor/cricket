# cricket

Designing a machine learning powered cricket app, that is designed to imitate [WinViz](https://cricviz.com/winviz/).

Currently, only extracts raw data from [cricsheet](https://cricsheet.org/) and stores it in a ball-by-ball format in a JSONL file.

## Usage

Firstly, clone the repository and navigate to the directory.

Then, if not already done, install Poetry using the intructions [here](https://python-poetry.org/docs/#installation).

Now you can install all the dependencies using the following command:

```bash
poetry install
```

And then activate the virtual environment using:

```bash
poetry shell
```

To run the current script, firstly download the (JSON) data from [cricsheet](https://cricsheet.org/matches/) and extract the zipped data into a folder named `input_data`.

```bash
python main.py
```

This will create files named `all_ball_by_ball.jsonl` and `all_match_metadata.jsonl` in a `data` folder (that will be created if it doesn't exist).

## To-Do

- Create a machine learning model to predict the score of a team (with variance so we can get a range of scores) at the end of an over.
- Create methods to understand the game state of a match (and allow a user to input this).
- Create methods to predict the outcome of a game using the state and the model predictions.
- Refactor the data processing code to reduce the amount of modules required.
