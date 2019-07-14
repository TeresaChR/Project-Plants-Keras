from keras.applications import inception_v3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from base.base_model import BaseModel
import os.path as path
import tensorflow as tf
import keras.backend as K

class Inceptionv3_model(BaseModel):
    def __init__(self, config):
        super(Inceptionv3_model, self).__init__(config)
        self.build_model()

    def build_model(self):
        
        # create the base pre-trained model
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False,pooling=None)
        
        #for i,layer in enumerate(self.model.layers):
        #    print(i,layer.name)
        #base_model.summary()
        
        x=base_model.output
        x=GlobalAveragePooling2D(name='avg_pool')(x)
        preds=Dense(self.config.model.classes_num, activation='softmax', name='predictions')(x) #final layer with softmax activation

        self.model=Model(inputs=base_model.input,outputs=preds)

        self.model.summary()
            
            # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer=self.config.model.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
    #https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    def get_model_memory_usage(self, batch_size, model):
        import numpy as np
        from keras import backend as K
    
        shapes_mem_count = 0
        for l in model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem
    
        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    
        number_size = 4.0
        if K.floatx() == 'float16':
             number_size = 2.0
        if K.floatx() == 'float64':
             number_size = 8.0
    
        total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 6)
        return gbytes
    
 