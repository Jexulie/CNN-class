
from pprint import pprint as pp
import os
import argparse

from data import DataPreparer, ShapePreprocessor, ImageToArrayPreprocessor
from model import CNN_Model

#- CLI Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="./datasets/", required=False)
# ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

def defineCategories(path):
    # get subfolder count and names
    folderNames = os.listdir(path)
    return folderNames, len(folderNames)

#- Options
categories, classes = defineCategories(args['dataset'])
width, height, depth = 32, 32, 3
learning_rate = 0.005 # 0.005
batch_size = 32
epochs = 160 # 100
trainOps = {
    'batch_size': batch_size,
    'epochs': epochs
}

imgProps = {
    'width': width,
    'height': height,
    'depth': depth,
    'classes': classes
}

sp = ShapePreprocessor(width, height)
ita = ImageToArrayPreprocessor()

loader = DataPreparer(args["dataset"], preprocessors=[sp, ita])
trainX, testX, trainY, testY = loader.prepare(verbose=100)

model = CNN_Model()
model.prepareModel((trainX, testX, trainY, testY), trainOps, imgProps)

model.SaveModel('phones')

model.evaluate(batch_size, categories)
model.plot()