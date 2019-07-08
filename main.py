from comet_ml import Experiment
#from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
from data_loader.image_generator_data_loader import ImageGeneratorDataloader

#from models.simple_mnist_model import SimpleMnistModel
from utils import factory
from trainers.image_trainer import ImageTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from evaluater.image_evaluater import ImageEvaluater


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        
    except:
        print("missing or invalid arguments")
        exit(0)

  
    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    #data_loader = SimpleMnistDataLoader(config)
    data_loader = ImageGeneratorDataloader(config)
     
    print('Create the model.')
    model = factory.create("models."+config.model.name)(config)
    
    if args.test:       
        print ('Start testing')
        ImageEvaluater(model.model, data_loader.get_validation_generator(),data_loader.get_train_generator().class_indices, config)
    else:     
        print('Create the trainer')
        trainer = ImageTrainer(model.model, data_loader.get_train_generator(),data_loader.get_validation_generator(), config)
        
        print('Start training the model.')
        trainer.train()

          
if __name__ == '__main__':
    main()
