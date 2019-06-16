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

#WEIGHTS_PATH = "https://github.com/scheckmedia/keras-shufflenet/raw/master/weights/ShuffleNet_1X_g3_br_0.25_373.hdf5"
WEIGHTS_PATH ="/home/stephen/Downloads/ShuffleNet_1X_g3_br_0.25_373.hdf5"
#Code sourse if from, Need to review aginst paper
#https://github.com/scheckmedia/keras-shufflenet/blob/master/shufflenet.py


class ShuffleNetV2_model(BaseModel):

    def __init__(self, config):
        super(ShuffleNetV2_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        self.model = self.ShuffleNetV2(self)

        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.config.model.optimizer,
                      metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)
     
    # Modular function for Fire Node
    
    def ShuffleNetV2(self,include_top=True, input_tensor=None, weights='imagenet', scale_factor=1.0, pooling='avg',
                   input_shape=(224,224,3), groups=3, num_shuffle_units=[3, 7, 3],
                   bottleneck_ratio=0.25, classes=1000):
        
        raise NotImplementedError