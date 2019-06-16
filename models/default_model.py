from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation

class DefaultModel(BaseModel):
    def __init__(self, config):
        super(DefaultModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(self.config.trainer.dim, self.config.trainer.dim, 3)))
            
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
            
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
            
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
            
        self.model.add(Flatten())  # go from 3D to 1D
        self.model.add(Dense(64))  # Fully connected layer
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))  # dropout to avoid over-fitting from: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        self.model.add(Dense(self.config.model.classes_num))  # fully connected output layer
        self.model.add(Activation('softmax'))  # softmax the output within the range of (0 to 1) for prediction capabilities



        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.config.model.optimizer,
                      metrics=['accuracy']) # want to see how accurate the model is (but are minimize the loss)