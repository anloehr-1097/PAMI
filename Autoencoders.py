"""This is a implementation of Variational Autoencoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from future import *

def main():
    """main function"""
    print("Hello world")
    return None



class Encoder_fc(nn.Module):
    """Encoder model for VAE"""
    
    def __init__(self):
        super().__init__()  # inherit from nn.Module

        # input image dimensions: 600x600x3
        self.fc1 = nn.Linear(600 * 600 * 3, 1200)  
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder_fc(nn.Module):

    def __init__(self):
        super().__init__()  # inherit from nn.Module

        # input image dimensions: 600x600x3
        self.fc1 = nn.Linear(600 * 600 * 3, 1200)  
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VAE(nn.Module):

    def __init__(self):
        super().__init__()  # inherit from nn.Module

        # input image dimensions: 
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# only serves as a parent class to concrete distributions 
class DistributionType: ...

# one of which is a Gaussian distribution
class GaussianDistribution(DistributionType):
    def __init__(self, shape : tuple, mu : torch.Tensor, sigma : torch.Tensor) -> None:
        super().__init__()
        
        assert shape[0] == mu.shape[0], "wrong shape mu"
        assert shape[0] == sigma.shape[0], "wrong shape sigma"
        if not len(sigma.shape) == 1:
            assert sigma.shape[0] == sigma.shape[1], "expected sigma to be a matrix"

        self.mu = mu
        self.sigma = sigma
        self.shape = shape 


    def __call__(self) -> torch.Tensor:
        # get n samples from distribution 
        param_ind_dist = D.multivariate_normal.MultivariateNormal(
            loc=torch.Tensor(np.zeros(self.shape[0])),
            covariance_matrix=torch.Tensor(np.eye(self.shape[0])))


        # param_ind_dist = D.Normal(torch.Tensor(np.zeros(self.shape[0])), torch.Tensor(np.eye(self.shape[0]))) 
        sample = param_ind_dist.rsample()
        sample = torch.unsqueeze(sample, 1)
        print(sample)
        print(self.sigma)

        # assert sample.shape[0] == self.mu.shape[0], f"Shape mismatch: Shape of sample={sample.shape[0]}, Shape of mu={self.mu.shape[0]}"
        return self.sigma @ sample + self.mu


    def generate(self, shape : tuple, mu : torch.Tensor, sigma : torch.Tensor):

        super().__init__()
        
        assert shape[0] == mu.shape[0], "wrong shape mu"
        assert shape[0] == sigma.shape[0], "wrong shape sigma"
        if not len(sigma.shape) == 1:
            assert sigma.shape[0] == sigma.shape[1], "expected sigma to be a matrix"

        self.mu = mu
        self.sigma = sigma
        return None



class BernoulliDistribution(DistributionType): ...


class Encoder: 
    """General Encoder Class to be used in VAEs.
    
    The encoder acts on the inputs x, is parametrized by phi and produces a probability
    distribution over z  (conditioned on x).


    The encoder typically delivers the 'parameters' for some paramterized distribution
    which is used to sample z|x. (q_{phi}(z|x) in the paper)

    """

    def __init__(self, distr_type: DistributionType, num_latent : int) -> None:
        super.__init__()
        # define Model arch
        self.num_latent = num_latent
        self.f1 = nn.Linear(800*800, 784)
        self.f2 = F.relu()
        self.f3 = nn.Linear(784, 2*num_latent)  # one for mu one for sigma for each latent
        self.distr = distr_type  # must feature the __call__ method

    def forward(self, x : torch.Tensor):
        assert x.shape[0] == 800**2
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)

        # sampling

        mu = x[:self.num_latent]
        sigma = x[self.num_latent:]
        self.distr = self.distr.generate(shape=(self.num_latent, 1), mu=self.mu, sigma=self.sigma)
        return self.distr()

class Decoder:
    """General Decoder Class to be used in VAEs"""
    def __init__(self, distr_type: DistributionType, num_latent : int) -> None:
        super.__init__()
        # define Model arch
        self.num_latent = num_latent
        self.f1 = nn.Linear(10, 784)
        self.f2 = F.tanh()
        self.f3 = nn.Linear(784, 2*800*800)  # one for mu one for sigma for each latent
        self.distr = distr_type  # must feature the __call__ method

    def forward(self, x : torch.Tensor):
        assert x.shape[0] == 10
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)

        # sampling

        mu = x[:800*800]
        sigma = x[800*800:]
        self.distr = self.distr.generate(shape=(800*800, 1), mu=mu, sigma=sigma)
        return self.distr()
        pass


class VAE:
    def __init__(self, enc: Encoder, dec: Decoder, prior, loss_fun) -> None:

        pass

def variational_lower_bound(x : torch.Tensor, enc : Encoder, dec: Decoder) -> torch.Tensor:
    """Custom Loss function used when training VAE Model.
     
    Implements an estimator of the ELBO as proposed in the paper
    'Auto-encoding Variational Bayes' by Kingma and Welling.
    """


    pass


