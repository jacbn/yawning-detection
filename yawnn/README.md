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

    - `pip install` the following modules:
    - required: `numpy scipy tensorflow`
    - (for plots) `matplotlib pyqt5 visualkeras`
    - (for testing and alternative classifiers) `scikit-learn`
    - `cd yawnn/` (if not here already)
    - `pip install -e .`

4. Note that `visualkeras 0.0.2` has a bug for plotting LSTMs, to fix:

    - Go to `.venv/lib/python3.9/site-packages/visualkeras/layered.py`
    - Comment out line 100, `z = min(max(z), max_z)`

### Running

Main code can be found in `./yawnnlib/`. `./yawnnlib/main.py` contains the highest-level functions for training on a given dataset.

`./tools/` contains some scripts for preprocessing data.
