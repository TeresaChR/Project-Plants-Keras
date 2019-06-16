from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from base.base_model import BaseModel
import os.path as path

class Inceptionv3_model(BaseModel):
    def __init__(self, config):
        super(Inceptionv3_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)
        
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have x classes
        predictions = Dense(self.config.model.classes_num, activation='softmax')(x)
        
        # this is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        
        if path.exists(self.config.model.pretrained_weights_file):

            print("----------Weight being loaded from file------------")
            self.model.load_weights(self.config.model.pretrained_weights_file)
            
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            # we chose to train the top 2 inception blocks, i.e. we will freeze
            # the first 172 layers and unfreeze the rest:
            for layer in self.model.layers[:172]:
                layer.trainable = False
            for layer in self.model.layers[172:]:
                layer.trainable = True
            
            # we need to recompile the model for these modifications to take effect
            # we use SGD with a low learning rate
            from keras.optimizers import SGD
            self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

            
        else:
            print("----------No Weight File ------------")
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False
            
            # compile the model (should be done *after* setting layers to non-trainable)
            self.model.compile(optimizer=self.config.model.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])