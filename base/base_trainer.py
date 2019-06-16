class BaseTrain(object):
    def __init__(self, model, trainData, valData, config):
        self.model = model
        self.trainData = trainData
        self.valData =  valData
        self.config = config

    def train(self):
        raise NotImplementedError
