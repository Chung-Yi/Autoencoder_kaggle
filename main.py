from data_loader.base_dataloader import BaseDataLoader
from configuration import ConfigParser
from trainer.trainer import Trainer

def main():
    config_parser = ConfigParser("config.ini")
    data_loader = BaseDataLoader(config_parser.training_data_path, config_parser.testing_data_path, config_parser.params)
    print("train data shape: ", data_loader.train_data.shape)
    print("test data shape: ", data_loader.test_data.shape)

    # model 
    trainer = Trainer(data_loader.train_dataloader, config_parser.params)
    trainer.train()


if __name__ == "__main__":
    main()