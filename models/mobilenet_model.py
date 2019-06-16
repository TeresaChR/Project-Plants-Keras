from base.base_model import BaseModel
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import mobilenet

class MobileNet_model(BaseModel):
    def __init__(self, config):
        super(MobileNet_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        self.model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.config.model.optimizer,
                      metrics=['accuracy']) # want to see how accurate the model is (but are minimize the loss)
        
        
        
