from yawnnlib.evaluation import metrics
from yawnnlib.utils import commons, filters, config
from yawnnlib.neural.fftModelInput import FFTModelInput
from yawnnlib.neural.eimuModelInput import EimuModelInput
from yawnnlib.training.models import MODEL_INPUTS

from yawnnlib.neural.modelInput import ModelInput

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
    from collections import defaultdict
    colorMap = defaultdict(dict)
    colorMap[tf.keras.layers.Dropout]['fill'] = '#e53d65' # red
    colorMap[tf.keras.layers.Reshape]['fill'] = '#aaffff' # light blue
    colorMap[tf.keras.layers.Conv2D]['fill'] = '#90ee90'  # light green
    colorMap[tf.keras.layers.LSTM]['fill'] = '#ffd166'    # light yellow
    colorMap[tf.keras.layers.Flatten]['fill'] = 'pink'
    colorMap[tf.keras.layers.Dense]['fill'] = 'purple'
    colorMap[tf.keras.layers.MaxPooling2D]['fill'] = 'orange'
    
    vk.layered_view(model, spacing=50, max_xy=800, draw_volume=True, to_file="model.png", legend=True, color_map=colorMap).show() # type: ignore

def testDataOnModel(model, modelType : ModelInput, dataDirectory : str) -> None:
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
    modelType = MODEL_INPUTS['eimuLSTM']
    model = loadModel(f"{MODELS_PATH}/eimuLSTM_13.h5")
    visualizeModel(model)
    # testDataOnModel(model, modelType, f"{DATA_PATH}/")