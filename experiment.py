import gpytorch
import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from data_mgmt import concrete_data_fetch, mnist_data_fetch

