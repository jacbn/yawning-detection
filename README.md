# yawning-detection

A third-year machine learning project to detect yawns using the IMU inside a pair of [eSense in-ear headphones](https://esense.io/).

## Description

This project is a feasibility study into detecting yawns from an ear-based wearable sensor. The headphones are equipped with a 6-axis IMU ([Inertial Measurement Unit](https://en.wikipedia.org/wiki/Inertial_measurement_unit)), which can record data at a rate up to 100Hz. The headphones connect to a mobile device via Bluetooth.

There are two main parts to the project: the mobile app, YeIMU (Flutter/Dart), used to interface with and obtain data from the headphones, and the data analyser, YawNN (Python), used to train and test models.

## Project Structure

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
│       │   
│       ├───evaluation/                                 # Evaluation code
│       │       metrics.py                                  # Definitions of the metrics used
│       │       test_model.py                               # Tests a model on a dataset
│       │   
│       ├───neural/                                       # Helper classes encapsulating the data for NNs
│       │       eimuLSTM.py                                 # LSTM for raw headphone data
│       │       fftCNN.py                                   # CNN for FFT data
│       │       modelType.py                                # Abstract parent class for helpers
│       │       spectrogramCNN.py                           # CNN for spectrogram data      
│       │           
│       ├───other_classifiers/                          # Non-NN classifiers
│       │       alternativeClassifiers.py                   # Manager for non-NN classifiers
│       │       knn.py                                      # K-Nearest Neighbours
│       │       svm_sk.py                                   # Support Vector Machine
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
