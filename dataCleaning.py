import glob
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split


## Code require to clean the DigitData for the DL Model

class FontImage:
    def __init__(self):
        self.dict = None
    
    def getImageDict(self):
        folderNames = glob.glob('DigitData/*')
        digitImageFilePaths = [glob.glob(folder + "/*.png") for folder in folderNames]

        ### Create ImageDict
        imgDict = {i: None for i in range(1, 10)}
        for k in imgDict:
            imgDict[k] = [cv2.imread(fpath) for fpath in digitImageFilePaths[k-1]]

        ## Convert Images for Greyscale and resize
        for k, v in imgDict.items():
            gray = [cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) for arr in v]
            imgDict[k] = [cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA) for arr in gray]
            # Add "channels" dimension
            imgDict[k] = np.expand_dims(imgDict[k], -1)
        
        self.dict = imgDict

    def getFontArrays(self):
        # Create x array with all image data
        x = np.array([v for v in self.dict.values()])
        x = np.reshape(x, newshape=(-1, 28, 28, 1))
        # Create label array
        y = np.array([np.repeat(k, len(self.dict[k])) for k in self.dict])
        y = np.reshape(y, newshape=(-1, 1))
        # Convert y to one-hot labels. Exclude zeros - invalid sudoku entry
        y = keras.utils.to_categorical(y, num_classes=10)[:, 1:]

        # Split into train, validation and test sets with shuffle
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.15,
                                                            shuffle=True,
                                                            random_state=0)
        
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.18,
                                                        shuffle=True,
                                                        random_state=33)
        
        # Invert images so backgrounds are black and fonts are white
        x_train = np.array(list(map(cv2.bitwise_not, x_train)))
        x_val = np.array(list(map(cv2.bitwise_not, x_val)))
        x_test = np.array(list(map(cv2.bitwise_not, x_test)))
        
        # Scale images to [0, 1]
        x_train = x_train.astype("float32") / 255
        x_val = x_val.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        
        # Add the "channels" dimension
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)
        x_test = np.expand_dims(x_test, -1)

        return x_train, x_val, x_test, y_train, y_val, y_test