import numpy as np
import random
import torch
from .dataset import CustomTensorDataset
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

class BaseDataLoader:
    def __init__(self, train_data_path, test_data_path, params):
        self.train_data = np.load(train_data_path)
        self.test_data = np.load(test_data_path)
        self.train_dataset = CustomTensorDataset(torch.from_numpy(self.train_data))
        self.test_dataset = CustomTensorDataset(torch.from_numpy(self.test_data))
        
        train_sampler = RandomSampler(self.train_dataset)
        test_sample = SequentialSampler(self.test_dataset)
        self.train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=int(params["batch_size"]))

        self.same_seeds()
    
    
    def same_seeds(self, seed=48763):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
           

        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True