from base.base_data_loader import BaseDataLoader
from keras.preprocessing.image import ImageDataGenerator


class ImageGeneratorDataloader(BaseDataLoader):
        
    def __init__(self, config):
        super(ImageGeneratorDataloader, self).__init__(config)
                                
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # regularise RGB channels (doesn't affect the actual color)
            zoom_range=[.65, 1],  # we are zooming in at most 35%, so in the range of 0 to 35%
            horizontal_flip=True,  # flipping the image horizontally
            brightness_range=[0.2,1.0], #brightness image augmentation randomly darken the image between 1.0 (no change) and 0.2 or 20%
            fill_mode='constant')

        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            self.config.data_loader.train_data_path,
            target_size=(self.config.trainer.dim, self.config.trainer.dim),  # resize train image to dim * dim
            batch_size=self.config.trainer.batch_size,
            class_mode='categorical',  # class mode is categorical - list with n-classes that points to the class it is
            shuffle = "false")
        self.validation_generator = test_datagen.flow_from_directory(
            self.config.data_loader.validation_data_path,
            target_size=(self.config.trainer.dim, self.config.trainer.dim),  # resize validation image to dim * dim
            batch_size=self.config.trainer.batch_size,
            class_mode='categorical',  # class mode is categorical - list with n-classes that points to the class it is
            shuffle = "false") 
        
        print(self.validation_generator.class_indices)  # allows us to see where the model will point to each class
        
    def get_train_generator(self):        
        return self.train_generator 

    def get_validation_generator(self):
        return self.validation_generator

