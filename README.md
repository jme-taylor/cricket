# Cricket

Designing a machine learning powered cricket app, that is designed to imitate [WinViz](https://cricviz.com/winviz/).

Currently, only extracts raw data from [cricsheet](https://cricsheet.org/) and stores it in a ball-by-ball format in a JSONL file.

## Current Progress

As of now, the data parsing has been done to turn the JSON data into a ball-by-ball parquet. This is done using [polars](https://pola.rs/). The next steps on this are to start building the logic to understand the game state and then to look into building the machine learning models.

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

```bash
python main.py
```

This will create files named `ball_by_ball.parquet` and `match_metadata.parquet` in a `data` folder (that will be created if it doesn't exist).

* `ball_by_ball.parquet` contains the ball-by-ball data, and various columns about that ball.
* `match_metadata.parquet` contains the metadata about the match, such as the teams, the venue, the date, etc.

## Design

There are some deliberate design choices made in this project, that if I was in the *perfect* scenario, I wouldn't have done. These are:

* Building the data processing, game state and machine learning models in one project. This makes the project very large in terms of code. I did this as I'm running it locally and can't afford to have productionised services interacting with each other, which is what should really be done.

## To-Do

An unordered, incomplete list of things I'd like to do in this project. Some will likely never get done, and some are way more important than others.

* Create a machine learning model to predict the score of a team (with variance so we can get a range of scores) at the end of an over.
* Create methods to understand the game state of a match (and allow a user to input this).
* Create methods to predict the outcome of a game using the state and the model predictions.
* Refactor the data processing code to reduce the amount of modules required.
* Build more sad path tests.
* Use pytest paremetrize in the tests more often.
* Build logic to only process data in the `input_data` folder that hasn't been processed yet (will be a huge time saver).
