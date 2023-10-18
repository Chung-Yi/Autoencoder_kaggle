from torch import nn

class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 3, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
