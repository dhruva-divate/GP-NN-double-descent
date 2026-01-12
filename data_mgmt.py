import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
def concrete_data_fetch(test_size=0.2, random_state=69):
    """Fetches the Concrete Compressive Strength dataset from the UCI Machine Learning Repository.
       Args:
        test_size (float): test-train split ratio.
        random_state (int): Random seed for reproducibility.
       Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        metadata (dict): Dataset metadata.
    """
    # fetch dataset 
    concrete_compressive_strength = fetch_ucirepo(id=165) 
  
    # data (as pandas dataframes) 
    X = concrete_compressive_strength.data.features 
    y = concrete_compressive_strength.data.targets.values.ravel() #becomes a 1d array
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Now standartise features for GPs
    # as GPs are sensitive to feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    metadata={'name': 'Concrete Compressive Strength',
              'n_samples': len(X),
              'n_features': X.shape[1],
              'description': concrete_compressive_strength.metadata.get('description', 'No description available.')}
    
    return X_train, X_test, y_train, y_test, metadata

from sklearn.datasets import fetch_openml

from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split

def mnist_data_fetch(digit_pairs=(3, 8), test_size=0.2, random_state=69, max_samples=None):
    """Fetches the MNIST dataset using torchvision.
       Args:
        digit_pairs (tuple): Two digits to use for binary classification. (3 and 8 by default, as they are often confused)
        test_size (float): test-train split ratio.
        random_state (int): Random seed for reproducibility.
        max_samples (int, optional): Maximum number of samples to use (for computational efficiency).
    Returns:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Training labels (0 or 1).
        y_test (np.ndarray): Test labels (0 or 1).
        metadata (dict): Dataset metadata.
    """
    # fetch dataset from torchvision
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)
    
    # Combine train and test, then we split it ourselves below
    X = np.concatenate([mnist_train.data.numpy(), mnist_test.data.numpy()])
    y = np.concatenate([mnist_train.targets.numpy(), mnist_test.targets.numpy()])
    
    # Flatten images (28x28 -> 784)
    X = X.reshape(-1, 784)
    
    # Filter for the specified digit pairs
    digit_0, digit_1 = digit_pairs
    filter_mask = (y == digit_0) | (y == digit_1)
    X = X[filter_mask]
    y = y[filter_mask]
    y = (y == digit_1).astype(int)  # Convert to binary labels (0 and 1)
    
    # subsampling
    if max_samples is not None and len(X) > max_samples:
        np.random.seed(random_state)
        indices = np.random.choice(len(X), size=max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # normalisation
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    metadata = {
        'name': f'MNIST Binary ({digit_0} vs {digit_1})',
        'n_samples': len(X),
        'n_features': X_train.shape[1],
        'digit_pair': digit_pairs
    }
    
    return X_train, X_test, y_train, y_test, metadata

def generate_synthetic_gp_data(n_samples=500, n_features=1, noise_std=0.1, 
                                 kernel_lengthscale=1.0, kernel_variance=1.0,
                                 test_size=0.2, random_state=42):
    """Generates synthetic regression data from a Gaussian Process prior.
    
    Args:
        n_samples (int): Total number of samples to generate.
        n_features (int): Dimensionality of input features.
        noise_std (float): Standard deviation of observation noise.
        kernel_lengthscale (float): Lengthscale parameter for RBF kernel.
        kernel_variance (float): Variance parameter for RBF kernel.
        test_size (float): Proportion of dataset to include in test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Training targets (noisy observations).
        y_test (np.ndarray): Test targets (noisy observations).
        f_true (np.ndarray): True function values (noiseless) for all X.
        metadata (dict): Dataset metadata.
    """
    np.random.seed(random_state)
    
    # Generate random input points
    X = np.random.uniform(-5, 5, size=(n_samples, n_features))
    
    # Compute RBF kernel matrix
    from scipy.spatial.distance import cdist
    distances = cdist(X, X, 'sqeuclidean')
    K = kernel_variance * np.exp(-distances / (2 * kernel_lengthscale**2))
    
    # Sample from GP prior
    f_true = np.random.multivariate_normal(np.zeros(n_samples), K)
    
    # Add observation noise
    y = f_true + np.random.normal(0, noise_std, n_samples)
    
    # Train-test split
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X, y, f_true, test_size=test_size, random_state=random_state
    )
    
    metadata = {
        'name': 'Synthetic GP Data',
        'n_samples': n_samples,
        'n_features': n_features,
        'noise_std': noise_std,
        'kernel_lengthscale': kernel_lengthscale,
        'kernel_variance': kernel_variance
    }
    
    return X_train, X_test, y_train, y_test, np.concatenate([f_train, f_test]), metadata


# Example usage
if __name__ == "__main__":
    np.random.seed(69)
    
    # Load concrete data
    X_train, X_test, y_train, y_test, meta = concrete_data_fetch()
    print(f"Concrete: {meta['n_samples']} samples, {meta['n_features']} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # Load MNIST binary classification
    X_train, X_test, y_train, y_test, meta = mnist_data_fetch(digit_pairs=(3, 8), max_samples=3000)
    print(f"MNIST: {meta['n_samples']} samples, {meta['n_features']} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # Generate synthetic data
    X_train, X_test, y_train, y_test, f_true, meta = generate_synthetic_gp_data(n_samples=500)
    print(f"Synthetic: {meta['n_samples']} samples, {meta['n_features']} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
