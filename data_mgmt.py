import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
def concrete_data_fetch():
    """Fetches the Concrete Compressive Strength dataset from the UCI Machine Learning Repository.
       Returns:
        X (pd.DataFrame): Features of the dataset.
        y (pd.Series): Target variable of the dataset.
        concrete_compressive_strength: Full dataset object with metadata.
    """
    # fetch dataset 
    concrete_compressive_strength = fetch_ucirepo(id=165) 
  
    # data (as pandas dataframes) 
    X = concrete_compressive_strength.data.features 
    y = concrete_compressive_strength.data.targets 
    return X, y, concrete_compressive_strength
    # metadata 
    # print(concrete_compressive_strength.metadata) 

    # variable information 
    # print(concrete_compressive_strength.variables)

def mnist_data_fetch():
    """Fetches the MNIST dataset from the UCI Machine Learning Repository.
       Returns:
        X (pd.DataFrame): Features of the dataset.
        y (pd.Series): Target variable of the dataset.
        mnist: Full dataset object with metadata.
    """
    # fetch dataset 
    mnist = fetch_ucirepo(id=554) 
  
    # data (as pandas dataframes) 
    X = mnist.data.features 
    y = mnist.data.targets 
    return X, y, mnist
    # metadata 
    # print(mnist.metadata) 

    # variable information 
    # print(mnist.variables)

# Now considering synthetic datasets
"""Also have to consider making test/train splits for all datasets here.
   Would need to make sure to care about GP feature scaling and MNIST pixel value scaling and sampling for running GP."""

