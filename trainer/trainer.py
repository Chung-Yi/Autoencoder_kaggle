import torch
import numpy as np
from model import VAE, ConvAutoencoder, FcnAutoencoder
from torch import nn
from tqdm import tqdm
class Trainer:
    def __init__(self, train_dataloader, params):
        self.model_type = params["type"]
        self.params = params
        # self.num_epochs = int(params["num_epochs"])
        self.device = torch.device(params["device"])
        self.train_dataloader = train_dataloader
        self.model = None
        if self.model_type == "vae":
            self.model = VAE().to(self.device)
        elif self.model_type == "fcn":
            self.model = FcnAutoencoder().to(self.device)
        else:
            self.model = ConvAutoencoder().to(self.device)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train(self):

        epochs = int(self.params["num_epochs"])

        best_loss = np.inf
        self.model.train()

        epochs_iterator = tqdm(range(epochs))

        for epoch in epochs_iterator:
            total_loss = list()
            for data in self.train_dataloader:
                # loading
                img = data.float().to(self.device)

                if self.model_type == "fcn":
                    img = img.view(img.shape[0], -1)

                # forward
                output = self.model(img)
                if self.model_type == "vae":
                    loss = self.model.loss_vae(output[0], img, output[1], output[2], self.loss)
                else:
                    loss = self.loss(output, img)
                
                total_loss.append(loss.item())

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # save best
            
            mean_loss = np.mean(total_loss)
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(self.model, "save/best_model_{}.pt".format(self.model_type))
            
            # log
            epochs_iterator.set_description(
                'Train_Batch {} | mean loss: {:.5f}'.format(epoch + 1, mean_loss)
            )

            # save last
            torch.save(self.model, "save/last_model_{}.pt".format(self.model_type))






    

       