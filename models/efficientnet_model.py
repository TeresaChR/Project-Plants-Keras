from base.base_model import BaseModel
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D,GlobalMaxPooling2D, Input, Dense, DepthwiseConv2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.engine.topology import get_source_inputs

from keras.utils import get_file
from keras.utils import layer_utils
import numpy as np



#https://github.com/qubvel/efficientnet
from efficientnet import EfficientNetB0


#Code sourse if from, Need to review aginst paper



class Efficientnet_model(BaseModel):

    def __init__(self, config):
        super(Efficientnet_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        self.model = EfficientNetB0(weights='imagenet')

        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.config.model.optimizer,
                      metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)
     
    # Modular function for Fire Node
    