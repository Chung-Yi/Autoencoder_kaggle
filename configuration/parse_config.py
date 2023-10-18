import configparser

class ConfigParser:
    def __init__(self, config_path) -> None:
        self.config = configparser.ConfigParser()
        self.config.sections()
        self.config.read(config_path)
       
        print(self.config.sections())
        self.training_data_path = self.config["DATAPATH"]["TrainingDataPath"]
        self.testing_data_path = self.config["DATAPATH"]["TestingDataPath"]
        self.params = self.config["TraingParams"]