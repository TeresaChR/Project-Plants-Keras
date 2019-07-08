from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard


class ImageTrainer(BaseTrain):
    def __init__(self, model, trainData, valData, config):
        super(ImageTrainer, self).__init__(model, trainData, valData, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()
        

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if hasattr(self.config,"comet"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet.api_key, project_name=self.config.exp.name, workspace=self.config.comet.workspace)
            experiment.disable_mp()
            experiment.log_parameters(self.acc)
            self.callbacks.append(experiment.get_keras_callback())

        
    def train(self):  
        history=self.model.fit_generator(
            self.trainData,  # passes the training data through this to transform it
            steps_per_epoch=self.config.trainer.num_training_img // self.config.trainer.batch_size,  # how many times we are stepping for each epoch
            epochs=self.config.trainer.num_epochs,  # the from scratch model stopped improving after about 40 epochs
            verbose=self.config.trainer.verbose_training,
            validation_data=self.valData,  # passes the validation data through this
            validation_steps=self.config.trainer.num_val_img // self.config.trainer.batch_size,
            callbacks=self.callbacks,
        ) 
        
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
        
        self.model.summary() # print out summary of model for testing purposes