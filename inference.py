import torch
import os
import pandas as pd
from data_loader.base_dataloader import BaseDataLoader
from configuration import ConfigParser
from torch import nn


out_file = 'prediction.csv'

def main():
    config_parser = ConfigParser("config.ini")
    data_loader = BaseDataLoader(config_parser.training_data_path, config_parser.testing_data_path, config_parser.params)
    test_dataloader = data_loader.test_dataloader
    # load trained model
    model_path = "save/best_model_vae.pt"
    model = torch.load(model_path)
    model.eval()

    eval_loss = nn.MSELoss(reduction='none')

    anomaly = list()
    with torch.no_grad():
        device = torch.device("mps")
        for i, data in enumerate(test_dataloader):
            img = data.float().to(device)
            output = model(img)[0]
            loss = eval_loss(output, img).sum([1, 2, 3])
            print(loss)
            os._exit(0)
            anomaly.append(loss)
    anomaly = torch.cat(anomaly, axis=0)
    anomaly = torch.sqrt(anomaly).reshape(len(data_loader.test_data), 1).cpu().numpy()
            # if model_type in ['fcn']:
            #     img = img.view(img.shape[0], -1)
            #     output = model(img)
            # if model_type in ['vae']:
            #     output = output[0]
            # if model_type in ['fcn']:
            #     loss = eval_loss(output, img).sum(-1)
            # else:
            #     loss = eval_loss(output, img).sum([1, 2, 3])
    
    df = pd.DataFrame(anomaly, columns=['score'])
    df.to_csv(out_file, index_label = 'ID')

    



if __name__ == "__main__":
    main()