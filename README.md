# yawning-detection

A third-year machine learning project to detect yawns using the IMU inside a pair of [eSense in-ear headphones](https://esense.io/).

## Description

This project is a feasibility study into detecting yawns from an ear-based wearable sensor. The headphones are equipped with a 6-axis IMU ([Inertial Measurement Unit](https://en.wikipedia.org/wiki/Inertial_measurement_unit)), which can record data at a rate up to 100Hz. The headphones connect to a mobile device via Bluetooth.

There are two main parts to the project: the mobile app, YeIMU (Flutter/Dart), used to interface with and obtain data from the headphones, and the data analyser, YawNN (Python), used to train and test models.

## Project Structure

For the code used to generate the results used in the paper, see branch 'paper'.

Instructions for how to run each part are in the READMEs of their respective directories.

```text
.
├───yawnn/                                              # Yawn Analyser
│   │   README.md                                           # YawNN Readme
│   │   requirements.txt                                    # Python dependencies
│   │   
│   ├───data/                                           # Data files
│   │
│   ├───trained_models/                                 # Saved models
│   │ 
│   ├───test/                                           # Unit tests; directory structure mirrors yawnnlib
│   │       README.md                                       # Instructions for running tests
│   │               
│   ├───tools/                                          # Tools for data processing
│   │       eimuResampler.py                              # Resampler for raw headphone data
│   │       
│   └───yawnnlib/                                       # Main code
│       │   main.py                                         # Entry point; trains and tests models
│       │   config.yaml                                     # Global hyperparameter configuration
│       │   
│       ├───alternatives/                               # Classical ML approaches (non-NN)
│       │       alternative_classifier.py                   # Wrapper class for pipelined approach
│       │       manual/                                     # Manually-written approaches
│       │   
│       ├───evaluation/                                 # Evaluation code
│       │       metrics.py                                  # Definitions of the metrics used
│       │       test_model.py                               # Tests a model on a dataset
│       │   
│       ├───preprocessing/                              # Classes encapsulating the input to a classifier
│       │       eimuModelInput.py                           # Raw .eimu headphone data
│       │       fftModelInput.py                            # FFT of headphone data
│       │       modelInput.py                               # Abstract parent class
│       │       spectrogramModelInput.py                    # Spectrogram of headphone data
│       │ 
│       ├───structure/                                  # Classes representing the underlying data
│       │       fourierData.py                              # A Fourier-transformed dataset
│       │       sensorReading.py                            # A single sensor reading
│       │       sessionData.py                              # A standard dataset
│       │       timestamp.py                                # A timestamp
│       │
│       ├───training/                                   # Training code
│       │       models.py                                   # Defines the Keras models to be trained
│       │       trainingFuncs.py                            # Helper functions for training models
│       │
│       └───utils/                                      # Utility code
│               commons.py                                  # Common functions
│               config.py                                   # Interfaces with config.yaml
│               filters.py                                  # Data filters 
│
│
└───yeimu/                                              # Mobile app to interface with headphones
    │   README.md                                           # YawNN Readme
    │
    ├───{build target}/                                 # OS-specific build files
    │
    └───lib/                                            # Main code
        │   main.dart                                       # User-facing layout code
        │   results.dart                                    # Results page
        │
        ├───io/                                         # IO-related code
        │       io_manager.dart                             # Saving and loading data  
        │       sensor_data_file.dart                       # Converts session data to/from strings
        │
        ├───structure/                                  # Common classes
        │       sensor_data.dart                            # Individual sensor readings
        │       timestamps.dart                             # User-marked timestamps
        │
        └───versioning                                  # Versioning code
                version.dart                                Data-only class defining current version
```
