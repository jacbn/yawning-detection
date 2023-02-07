from yawnnlib.utils import commons, filters
from yawnnlib.neural.eimuLSTM import EimuLSTMInput
from yawnnlib.neural.fourierLSTM import FourierLSTMInput
from yawnnlib.neural.fourierCNN import FourierCNNInput
from yawnnlib.neural.modelType import ModelType

from tensorflow.keras.models import load_model

#todo: move to config
MODELS_PATH = f"{commons.PROJECT_ROOT}/models"
DATA_PATH = f"{commons.PROJECT_ROOT}/data"

def loadModel(modelPath : str):
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
    model = load_model(modelPath)
    model.summary()
    return model


def testDataOnModel(model, modelType : ModelType, dataDirectory : str):
    """ Tests the model on the data in a given directory. 
    
    Attributes
    ----------
    model : tensorflow.keras.models.X
        The tf model to test. Load a pre-trained model with loadModel(path).
    modelType : commons.ModelType
        The type of model to test.
    dataDirectory : str
        The directory containing the data to test.
    """
    annotatedData = modelType.fromDirectory(dataDirectory)
    _, (X, Y) = modelType.fromAnnotatedDataList(annotatedData, shuffle=True, equalPositiveAndNegative=False, trainSplit=0.0)
    model.evaluate(X, Y)

if __name__ == "__main__":
    # Main EimuLSTM on all data @ 96Hz
    modelType = EimuLSTMInput(dataFilter=filters.SmoothFilter(keepData=0.8), sessionGap=32)
    model = loadModel(f"{MODELS_PATH}/eimuLSTM_3.h5")
    testDataOnModel(model, modelType, f"{DATA_PATH}/tests/96hz/")