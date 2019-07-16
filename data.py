from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import cv2
import os
from imutils import paths
import numpy as np

#- Preprocessors

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat
    
    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)

class ShapePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

#- Data Preparer

class DataPreparer:
    def __init__(self, datasetFolder, preprocessors=None):
        self.__preprocessors = preprocessors

        self.__imagePaths = list(paths.list_images(datasetFolder))
        self.__imageNum = len(self.__imagePaths)

        if self.__preprocessors is None:
            self.__preprocessors = []

    #_ Loads | Preps Images returns matrix values
    def __dataloader(self, verbose=-1):
        print("[INFO] Loading Data...")

        data = []
        labels = []

        for (i, imagePath) in enumerate(self.__imagePaths):
            img = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2] 

            if self.__preprocessors is not None:
                for p in self.__preprocessors:
                    img = p.preprocess(img)
            
            data.append(img)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"processed {i + 1}/{self.__imageNum}")

        return (np.array(data), np.array(labels))
    

    def prepare(self, splitPercent=0.25, randomState=42, verbose=-1):
        print("[INFO] Preparing Images...")

        # load data
        data, labels = self.__dataloader(verbose)
        # transform data - labels
        data = data.astype("float") / 255.0

        LB = LabelBinarizer()

        # seperate data - test
        trainX, testX, trainY, testY = train_test_split(data, labels, test_size=splitPercent, random_state=randomState)

        trainY = LB.fit_transform(trainY)
        testY = LB.fit_transform(testY)

        return (trainX, testX, trainY, testY)