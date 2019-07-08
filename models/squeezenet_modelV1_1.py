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
from keras.layers import GlobalMaxPooling2D, Dense,GlobalAveragePooling2D
from keras.optimizers import SGD

WEIGHTS_PATH = 'https://github.com/wohlert/keras-squeezenet/releases/download/v0.1/squeezenet_weights.h5'


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

#Code sourse if from, Need to review aginst paper
#https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py


class SqueezeNetv1_1_model(BaseModel):

    def __init__(self, config):
        super(SqueezeNetv1_1_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        base_model =  self.SqueezeNetV1_1(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


                #for i,layer in enumerate(self.model.layers):
        #    print(i,layer.name)
        #base_model.summary()
        
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        preds=Dense(self.config.model.classes_num, activation='softmax')(x) #final layer with softmax activation
        
        self.model=Model(inputs=base_model.input,outputs=preds)
        
        #  This compiles the model architecture and the necessary functions that we
        #  categorical crossentropy is the loss function for classification problems with more than 2 classes
        self.model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        #self.model.compile(loss='categorical_crossentropy',
        #               optimizer=self.config.model.optimizer,
        #               metrics=['accuracy'])  # want to see how accurate the model is (but are minimize the loss)
       
    # Modular function for Fire Node

    def _fire(self,x, filters, name="fire"):
        sq_filters, ex1_filters, ex2_filters = filters
        squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
        expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
        expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
        x = Concatenate(axis=-1, name=name)([expand1, expand2])
        return x
    
        
    def SqueezeNetV1_1(self, include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):
    
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')
    
        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')
        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=48,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top)
    
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
    
        x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name='conv1')(img_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="valid")(x)
    
        x = self._fire(x, (16, 64, 64), name="fire2")
        x = self._fire(x, (16, 64, 64), name="fire3")
    
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="valid")(x)
    
        x = self._fire(x, (32, 128, 128), name="fire4")
        x = self._fire(x, (32, 128, 128), name="fire5")
    
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="valid")(x)
    
        x = self._fire(x, (48, 192, 192), name="fire6")
        x = self._fire(x, (48, 192, 192), name="fire7")
    
        x = self._fire(x, (64, 256, 256), name="fire8")
        x = self._fire(x, (64, 256, 256), name="fire9")
    

 
        
        model = Model(img_input, x, name="squeezenetv1_1")
        model.summary()
        
        if weights == 'imagenet':
            weights_path = get_file('squeezenet_weights.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
    
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
      

        return model
