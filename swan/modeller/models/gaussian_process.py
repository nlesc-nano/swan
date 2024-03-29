"""Module to Generate Gaussian processes models. See: https://docs.gpytorch.ai"""

import gpytorch as gp
from torch import Tensor


class GaussianProcess(gp.models.ExactGP):
    def __init__(
            self, train_x: Tensor, train_y: Tensor):
        likelihood = gp.likelihoods.GaussianLikelihood()
        super(GaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
