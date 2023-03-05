from yawnnlib.evaluation import metrics
from yawnnlib.utils import commons, filters, config
from yawnnlib.neural.fftCNN import FftCNNInput
from yawnnlib.neural.eimuLSTM import EimuLSTMInput
from yawnnlib.training.models import MODEL_INPUTS

from yawnnlib.neural.modelType import ModelType

import visualkeras as vk
import numpy as np
import tensorflow as tf

MODELS_PATH = config.get("MODELS_PATH")
DATA_PATH = config.get("DATA_PATH")
YAWN_TIME = config.get("YAWN_TIME")

def loadModel(modelPath : str) -> tf.keras.models.Model:
    """ Loads a pre-trained model.
    
    Attributes
    ----------
    modelPath : str
        The path to the model to load.
    
    Returns
    -------
    model : tensorflow.keras.models.X
        The loaded model.
    """
    model = tf.keras.models.load_model(modelPath)
    assert isinstance(model, tf.keras.models.Model), "Failed to load model."
    model.summary()
    return model

def visualizeModel(model : tf.keras.models.Model) -> None:
    vk.layered_view(model, spacing=50, max_xy=800, draw_volume=True, to_file="model.png").show() # type: ignore

def testDataOnModel(model, modelType : ModelType, dataDirectory : str) -> None:
    """ Tests the model on the data in a given directory. 
    
    Attributes
    ----------
    model : tensorflow.keras.models.Model
        The tf model to test. Load a pre-trained model with loadModel(path).
    modelType : commons.ModelType
        The type of model to test.
    dataDirectory : str
        The directory containing the data to test.
    """
    annotatedData = modelType.fromDirectory(dataDirectory)
    _, (X, Y) = modelType.fromAnnotatedDataList(annotatedData, shuffle=True, equalPositiveAndNegative=False, trainSplit=0.0)
    model.evaluate(X, Y)
    predY = np.round(model.predict(X)).astype(bool)
    metrics.evaluate(Y, predY)

if __name__ == "__main__":
    modelType = MODEL_INPUTS['fftCNN']
    model = loadModel(f"{MODELS_PATH}/fourierLSTM_3.h5")
    visualizeModel(model)
    testDataOnModel(model, modelType, f"{DATA_PATH}/")