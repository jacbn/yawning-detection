# YawNN

The machine-learning section of my project to detect yawns.

## How to use

### Requirements

- Python 3.8+

### Installation

1. Clone the repository.
2. (Optional, recommended) Create a virtual environment:

    > `python -m venv ./venv`
    > `. venv/bin/activate` (Linux) or `./venv/Scripts/activate` (Windows)
    > (To deactivate the virtual environment, run `deactivate`)

3. Install the dependencies in `./requirements.txt`:

    > `pip install -r requirements.txt`

### Running

Main code can be found in `./yawnnlib/`. `./yawnnlib/main.py` contains the highest-level functions for training on a given dataset.

`./tools/` contains some scripts for preprocessing data.
