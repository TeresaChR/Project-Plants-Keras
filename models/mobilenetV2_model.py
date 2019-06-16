from base.base_model import BaseModel
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import mobilenet_v2

class MobileNetV2_model(BaseModel):
    def __init__(self, config):
        super(MobileNetV2_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        self.model =  mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.config.model.optimizer,
                      metrics=['accuracy']) # want to see how accurate the model is (but are minimize the loss)
        
        
        
