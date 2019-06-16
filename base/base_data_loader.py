class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError
    
    def get_train_generator(self):
        raise NotImplementedError

    def get_validation_generator(self):
        raise NotImplementedError
