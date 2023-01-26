# YawNN

The machine-learning section of my project to detect yawns.

## How to use

### Requirements

- Python 3.9

### Installation

1. Clone the repository.
2. (Optional, recommended) Create a virtual environment:

    - `python -m venv ./venv`;
    - `. venv/bin/activate` (Linux) or `./venv/Scripts/activate` (Windows);
    - (run `deactivate` to exit the virtual environment when finished)

3. Either (1) install the dependencies in `./requirements.txt`:

    - `pip install -r requirements.txt`

   Or (2) perform a manual install:

    - `pip install numpy scipy tensorflow matplotlib pyqt5`
    - `cd yawnn/` (if not here already)
    - `pip install -e .`

### Running

Main code can be found in `./yawnnlib/`. `./yawnnlib/main.py` contains the highest-level functions for training on a given dataset.

`./tools/` contains some scripts for preprocessing data.
