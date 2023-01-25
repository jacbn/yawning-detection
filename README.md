# yawning-detection

A third-year machine learning project to detect yawns using the IMU inside a pair of [eSense in-ear headphones](https://esense.io/).

## Description

This project is a feasibility study into detecting yawns from an ear-based wearable sensor. The headphones are equipped with a 6-axis IMU ([Inertial Measurement Unit](https://en.wikipedia.org/wiki/Inertial_measurement_unit)), which can record data at a rate up to 100Hz. The headphones connect to a mobile device via Bluetooth.

There are two main parts to the project: the mobile app, YeIMU (Flutter/Dart), used to interface with and obtain data from the headphones, and the data analyser, YawNN (Python), used to train and test models.

## Project Structure

Instructions for how to run each project are in the READMEs of their respective directories.

```text
├── yeimu/                              # YeIMU mobile app
    ├── lib/                            # Dart code
        ├── io/                         # IO-related code
            ├── io_manager.dart         # Saving and loading data
            ├── sensor_data_file.dart   # Converting whole session data to/from strings
        ├── structure/                  # Common classes
            ├── sensor_data.dart        # Individual sensor readings
            ├── timestamps.dart         # Timestamps
        ├── versioning/                 # Versioning code
            ├── version.dart            # Data-only class defining current version
        ├── main.dart                   # Main app code
        ├── results.dart                # Results page

├── yawnn                               # YawNN data analyser
    ├── data/                           # Data files
    ├── lib/                            # Main code
        ├── classifiers/                # Non-NN classifiers
            ├── knn.py                  # K-Nearest Neighbours
            ├── svm_sk.py               # Support Vector Machine
        ├── alternativeClassifiers.py   Manager for non-NN classifiers
        ├── commons.py                  # Common functions
        ├── eimuLSTM.py                 # LSTM for raw headphone data
        ├── eimuReader.py               # Manager for raw headphone data
        ├── filters.py                  # Data filters 
        ├── fourierLSTM.py              # LSTM for Fourier-transformed data
        ├── fourierReader.py            # Manager for Fourier-transformed data
        ├── main.py                     # Entry point for YawNN
    ├── models/                         # Saved models
    ├── tools/                          # Tools for data processing
        ├── eimuDownsampler.py          # Downsampler for raw headphone data
```
