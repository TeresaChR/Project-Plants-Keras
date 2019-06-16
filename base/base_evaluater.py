class BaseEvaluater(object):
    def __init__(self, model, valData, train_labels, config):
        self.model = model
        self.valData =  valData
        self.train_labels = train_labels
        self.config = config

    def format_top_five(self):
        raise NotImplementedError

    def create_percentages(self):
        raise NotImplementedError
    
    def top_five(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError
