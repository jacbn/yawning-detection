from yawnnlib.evaluation import metrics
from yawnnlib.utils import commons, filters, config
from yawnnlib.preprocessing.fftModelInput import FFTModelInput
from yawnnlib.preprocessing.eimuModelInput import EimuModelInput
from yawnnlib.training.models import MODEL_INPUTS
from yawnnlib.preprocessing.modelInput import ModelInput
from yawnnlib.alternatives.alternative_classifier import AlternativeClassifier
import tools.eimuResampler as eimuResampler

from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from os import listdir
from PIL import ImageFont
import visualkeras as vk
import numpy as np
import tensorflow as tf
import pickle

MODELS_PATH = config.get("MODELS_PATH")
DATA_PATH = config.get("DATA_PATH")
TEST_PATH = config.get("TEST_PATH")
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
    colorMap[tf.keras.layers.TimeDistributed]['fill'] = 'blue'
    font = ImageFont.truetype("data/font/arial.ttf", 24, encoding="unic")
    
    vk.layered_view(model, spacing=50, max_xy=800, draw_volume=True, to_file="model.png", legend=True, font=font, color_map=colorMap).show() # type: ignore

def testDataOnModel(model, modelType : ModelInput, dataDirectory : str, isHafar : bool = False, resampleFrequency : int = -1):
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
    if isHafar:
        combinedTuple = modelType.fromHafarDirectory(dataDirectory)
        _, (X, Y) = modelType.fromCombinedTuple(combinedTuple, shuffle=True, equalPositiveAndNegative=True, trainSplit=0.0)
    else:
        annotatedData = modelType.fromEimuDirectory(dataDirectory)
        _, (X, Y) = modelType.fromAnnotatedDataList(annotatedData, shuffle=True, equalPositiveAndNegative=True, trainSplit=0.0)
    if resampleFrequency > 0:
        # todo: use frequency given by model as opposed to hardcoding
        (X, Y) = eimuResampler.resampleAnnotatedData((X, Y), 96, resampleFrequency)
    if "CNN-LSTM" in modelType.getType():
        (X, Y) = commons.timeDistributeAnnotatedData((X, Y))
    # res1 = model.evaluate(X, Y)
    Y = np.round(Y).astype(bool).squeeze()
    predY = np.round(model.predict(X)).astype(bool).squeeze()
    # print(list(zip(Y, predY)))
    res2 = metrics.evaluate(Y, predY)
    return res2 # accuracy, precision, recall, f1
    
def testDataOnAlternativeModels(altModelsPath : str, dataDirectory : str):
    
    ALT_MODELS_PATH = MODELS_PATH + "/alternative/"
    modelType = MODEL_INPUTS['altModels']
    annotatedData = modelType.fromEimuDirectory(dataDirectory)
    _, (testX, testY) = modelType.fromAnnotatedDataList(annotatedData, shuffle=True, equalPositiveAndNegative=True, trainSplit=0.0)
    fig = 1
    
    samples, windowSize, channels = testX.shape
    flattenedTestX = np.reshape(testX, (samples * windowSize, channels))
    extendedTestY = np.repeat(testY, windowSize)
    
    for file in listdir(ALT_MODELS_PATH):
        if file.endswith(".pkl"):
            with open(ALT_MODELS_PATH + file, "rb") as handler:
                classifier : AlternativeClassifier = pickle.load(handler)
                assert classifier.clf is not None, "Classifier not loaded."
                score = classifier.score(flattenedTestX, extendedTestY)
                plt.figure(fig)
                disp = ConfusionMatrixDisplay.from_estimator(classifier.clf, flattenedTestX, extendedTestY, display_labels=["No Yawn", "Yawn"])
                disp.ax_.set_title(classifier.name)
                print(f"{classifier.name} score: {score}")
                predY = np.round(classifier.clf.predict(flattenedTestX)).astype(bool)
                metrics.evaluate(extendedTestY, predY)
                fig += 1
                
    plt.show()

if __name__ == "__main__":
    modelType = MODEL_INPUTS['eimuLSTM']
    model = loadModel(f"{MODELS_PATH}/eimuLSTM_0.h5")
    visualizeModel(model)
    
    # testDataOnModel(model, modelType, f"{commons.PROJECT_ROOT}/data/user_trials/PRESENTATION/", isHafar=False)
    # testDataOnModel(model, modelType, config.get("HAFAR_PATH"), isHafar=True)
    
    testDataOnModel(model, modelType, f"{TEST_PATH}/", resampleFrequency=32)
    # testDataOnAlternativeModels(f"{MODELS_PATH}/alternative/", f"{TEST_PATH}/")