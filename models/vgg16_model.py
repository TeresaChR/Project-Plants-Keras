from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.applications import vgg16 

class Vgg16_model(BaseModel):
    def __init__(self, config):
        super(Vgg16_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        self.model =vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.config.model.optimizer,
                      metrics=['accuracy']) # want to see how accurate the model is (but are minimize the loss)