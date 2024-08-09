from .hyperparams import Hyperparams


class Tuner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def get_best_hyperparams(self, model, dataset) -> Hyperparams:
        ...
