# yawning-detection

A third-year machine learning project to detect yawns using the IMU inside a pair of [eSense in-ear headphones](https://esense.io/).

## Description

This project is a feasibility study into detecting yawns from an ear-based wearable sensor. The headphones are equipped with a 6-axis IMU ([Inertial Measurement Unit](https://en.wikipedia.org/wiki/Inertial_measurement_unit)), which can record data at a rate up to 100Hz. The headphones connect to a mobile device via Bluetooth.

There are two main parts to the project: the mobile app, YeIMU (Flutter/Dart), used to interface with and obtain data from the headphones, and the data analyser, YawNN (Python), used to train and test models.

## Project Structure

Instructions for how to run each project are in the READMEs of their respective directories.

```text
.
├───yawnn/
│   │   README.md                                       # YawNN Readme
│   │   requirements.txt                                # Python dependencies
│   │   
│   ├───data/                                           # Data files
│   │
│   ├───models/                                         # Saved models
│   │               
│   ├───tools/                                          # Tools for data processing
│   │       eimuResampler.py                              # Resampler for raw headphone data
│   │       
│   └───yawnnlib/                                       # Main code
│       │   main.py                                         # Entry point; trains and tests models
│       │   
│       ├───commons/                                    # Common code
│       │       commons.py                                  # Common functions
│       │       filters.py                                  # Data filters 
│       │           
│       ├───lstm/                                       # LSTM code
│       │       eimuLSTM.py                                 # LSTM for raw headphone data
│       │       fourierLSTM.py                              # LSTM for Fourier-transformed data
│       │           
│       ├───other_classifiers/                          # Non-NN classifiers
│       │       alternativeClassifiers.py                   # Manager for non-NN classifiers
│       │       knn.py                                      # K-Nearest Neighbours
│       │       svm_sk.py                                   # Support Vector Machine
│       │
│       └───readers/                                    # Data readers
│               eimuReader.py                               # Manager for raw headphone data
│               fourierReader.py                            # Manager for Fourier-transformed data
│
│
└───yeimu/
    │   README.md                                       # YawNN Readme
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
