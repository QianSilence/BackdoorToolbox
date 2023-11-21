import torch
import torch.nn as nn
class discriminator(nn.Module):
    
    """discriminator network.
    Args:
        z_dim (int): dimension of latent code (typically a number in [10 - 256])
        x_dim (int): for example m x n x c for [m, n, c]
    """
    def __init__(self, z_dim=2, x_dim=784):
        super(discriminator, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.net = nn.Sequential(
            nn.Linear(self.x_dim + self.z_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 400),
            nn.ReLU(True),
            nn.Linear(400, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
        )

    def forward(self, x, z):
        """
        Inputs:
            x : input from train_loader (batch_size x input_size )
            z : latent codes associated with x (batch_size x z_dim)
        """
        x = x.view(-1, self.x_dim)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()