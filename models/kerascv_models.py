import keras.backend as K
from base.base_model import BaseModel

from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalMaxPooling2D, Dense,GlobalAveragePooling2D, Flatten
from keras.optimizers import SGD

from keras.initializers import glorot_uniform

from kerascv.models import squeezenet
from kerascv.models import squeezenext
from kerascv.models import shufflenetv2b
from kerascv.models import shufflenet
from kerascv.models import common

class ShuffleNetV2(BaseModel):

    def __init__(self, config):
        super(ShuffleNetV2, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        #ShuffleNetV2 1x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,' https://arxiv.org/abs/1807.11164. 
        base_model =  shufflenetv2b.get_shufflenetv2b(width_scale=1.0, model_name="shufflenetv2b_w1",pretrained=True)

        ##shuffle delete 1 layers
        base_model.layers.pop()  
        #base_model.summary()
        x=base_model.layers[-1].output

        preds = Dense(units=self.config.model.classes_num, activation='softmax',name="output2")(x)
        self.model=Model(inputs=base_model.input,outputs=preds)
         
        #preds = common.flatten(x)
            #print(base_model.input)
       
        self.model.summary()
         
        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #self.model.summary()
        #self.model.compile(loss='categorical_crossentropy',
        #               optimizer=self.config.model.optimizer,
        #               metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)

class ShuffleNetV2_extra(BaseModel):

    def __init__(self, config):
        super(ShuffleNetV2_extra, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        #ShuffleNetV2 1x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,' https://arxiv.org/abs/1807.11164. 
        base_model =  shufflenetv2b.get_shufflenetv2b(width_scale=1.0, model_name="shufflenetv2b_w1",pretrained=True)

        ##shuffle delete 1 layers
        #base_model.layers.pop()  
        #base_model.summary()
         
        x=base_model.output

        preds = Dense(units=self.config.model.classes_num, activation='softmax',name="output2")(x)
        self.model=Model(inputs=base_model.input,outputs=preds)
         
        #preds = common.flatten(x)
            #print(base_model.input)
       
        #self.model.summary()
         
        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #self.model.summary()
        #self.model.compile(loss='categorical_crossentropy',
        #               optimizer=self.config.model.optimizer,
        #               metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)

class SqueezeNet_extra(BaseModel):

    def __init__(self, config):
        super(SqueezeNet_extra, self).__init__(config)
        self.build_model()

    def build_model(self):
        
         
        base_model =  squeezenet.get_squeezenet(version ='1.1',residual=False, model_name="squeezenet_v1_1",pretrained=True)
        x=base_model.output

        preds = Dense(units=self.config.model.classes_num, activation='softmax',name="output2")(x)
        self.model=Model(inputs=base_model.input,outputs=preds)
        self.model.summary()
         
        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #self.model.summary()
        #self.model.compile(loss='categorical_crossentropy',
        #               optimizer=self.config.model.optimizer,
        #               metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)
        
        
       
class SqueezeNet(BaseModel):

    def __init__(self, config):
        super(SqueezeNet, self).__init__(config)
        self.build_model()

    def build_model(self):
        
         
        base_model =  squeezenet.get_squeezenet(version ='1.1',residual=False, model_name="squeezenet_v1_1",pretrained=True)
       # base_model.summary()
        print("======================================================================")
            ##squeeze delete 4 layers
        base_model.layers.pop()
        base_model.layers.pop() 
        base_model.layers.pop()
        base_model.layers.pop()  
        #base_model.summary()
         
        x=base_model.layers[-1].output
 
        x = Convolution2D(filters=self.config.model.classes_num,kernel_size=1, name="output/final_conv")(x)
        x = Activation("relu", name="output/final_activ")(x)
        x = AveragePooling2D(pool_size=13, strides=1, name="output/final_pool")(x)
         
        preds = common.flatten(x)
            #print(base_model.input)
        self.model = Model(inputs=base_model.input, outputs=preds)
 
        self.model.summary()
        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #self.model.summary()
        #self.model.compile(loss='categorical_crossentropy',
        #               optimizer=self.config.model.optimizer,
        #               metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)
        
        
       
class Squeezenext_model(BaseModel):

    def __init__(self, config):
        super(Squeezenext_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        

        base_model =  squeezenext.get_squeezenext(version="23", width_scale=1.5, model_name="sqnxt23_w3d2",pretrained=True)
        base_model.summary()
        print("======================================================================")
            ##squeeze delete 4 layers
        base_model.layers.pop()  
        base_model.layers.pop()  
        
        x=base_model.layers[-1].output
        x=Flatten()(x)
        x=Dense(self.config.model.classes_num, activation='softmax')(x) #final layer with softmax activation
        base_model.summary()
            #print(base_model.input)
        self.model = Model(inputs=base_model.input, outputs=x)

        
        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(optimizer=self.config.model.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        #self.model.summary()

       
       
        #self.model.compile(loss='categorical_crossentropy',
        #               optimizer=self.config.model.optimizer,
        #               metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)
          
    # Modular function for Fire Node

    
        
    