from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import SGD #, Adam, Adamax
from keras.models import load_model
from keras import backend as B
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class CNN_Model:
    def build(self, width, height, depth, classes):
        print("[INFO] Building Model...")

        model = Sequential()
        inputShape = (height, width, depth)

        if B.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        #_ Layers
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        self._model = model

    def compile(self, isBinary, learningRate):
        print("[INFO] Compiling Model... ")
        
        optimizer = SGD(learningRate)
    
        if isBinary == True:
            lossFunc = "categorical_crossentropy"
        else:
            lossFunc = "binary_crossentropy"

        self._model.compile(loss=lossFunc, optimizer=optimizer, metrics=["accuracy"])

    def train(self, trainData, batchSize, epochs):
        print("[INFO] Training Model... ")
        self.X = trainData[0]
        self.x = trainData[1]
        self.Y = trainData[2]
        self.y = trainData[3]

        self.history = self._model.fit(self.X, self.Y, validation_data=(self.x, self.y), batch_size=batchSize, epochs=epochs, verbose=epochs/10)

    
    def SaveModel(self, fileName):
        print("[INFO] Saving Model... ")
        self._model.save(f'{fileName}.model')

    
    def LoadModel(self, fileName):
        self._model = load_model(fileName+'.model')
        return self._model

    
    def prepareModel(self, trainData, trainingOpts, imageProperties, learningRate=0.01):
        print("[INFO] Preparing Model... ")

        w = imageProperties['width']
        h = imageProperties['height']
        d = imageProperties['depth']
        c = imageProperties['classes']

        bSize = trainingOpts['batch_size']
        epochs = trainingOpts['epochs']
        self.epochs = epochs

        isBinary = False if c > 2 else True

        self.build(w, h, d, c)
        self.compile(isBinary, learningRate)
        self.train(trainData, bSize, epochs)

        return self._model

    def evaluate(self, batch_size, targetNames):
        print("[INFO] Evaluating Model... ")

        predictions = self._model.predict(self.x, batch_size)
        print(classification_report(self.y.argmax(axis=1), predictions.argmax(axis=1),target_names=targetNames))

    def plot(self, save=True):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), self.history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["acc"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        if save == True:
            plt.savefig("acc-loss.png")
        
        plt.show()