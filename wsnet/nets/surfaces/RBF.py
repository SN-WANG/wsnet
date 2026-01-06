# Radial Basis Function (RBF) Surrogate Model
# Author: Shengning Wang

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict, Any, Tuple, Union, Optional


# Define the type for the regression model object
RBFModel = Union[Ridge, LinearRegression]


class RBF_Transformer:
    """
    Transforming raw features X into RBF features Phi(X)
    """

    def __init__(self, num_centers: int = 100, gamma: Optional[float] = None, random_state: int = 42):
        self.num_centers = num_centers
        self.gamma = gamma
        self.centers = None
        self.random_state = random_state

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centers
        """

        num_samples = X.shape[0]
        pca = PCA(n_components=min(X.shape), random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        indices = np.linspace(0, num_samples - 1, self.num_centers, dtype=int)
        scores_selected = X_pca[indices, :]

        centers_init = pca.inverse_transform(scores_selected)

        return centers_init

    def fit(self, X: np.ndarray):
        """
        Find RBF centers using K-means and set gamma if 'scale' is desired.
        """

        # Determine Centers
        num_samples = X.shape[0]

        if self.num_centers >= num_samples:
            self.centers = X
            self.num_centers = num_samples
        else:
            centers_init = self._init_centers(X)
            kmeans = KMeans(n_clusters=self.num_centers, init=centers_init,
                            n_init=1, max_iter=500, random_state=self.random_state)
            kmeans.fit(X)
            self.centers = kmeans.cluster_centers_

        # Set gamma
        if self.gamma is None or self.gamma == 'scale':
            dist = euclidean_distances(self.centers, self.centers)
            np.fill_diagonal(dist, np.inf)
            min_dists = np.min(dist, axis=1)
            sigma = np.mean(min_dists)
            if sigma <= 1e-10:
                sigma = 1.0

            self.gamma = 1.0 / (2.0 * sigma ** 2)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the RBF transformation
        """

        # Calculate squared Euclidean distance between each point in X and each RBF center
        distances = euclidean_distances(X, self.centers, squared=True)

        # Apply the RBF Kernel (Gaussian)
        X_rbf = np.exp(-self.gamma * distances)

        return X_rbf

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit centers and transform the data
        """

        return self.fit(X).transform(X)


def train_rbf_model(X_train: np.ndarray, Y_train: np.ndarray, num_centers: int = 100,
                    gamma: Union[float, str] = 'scale', alpha: float = 1.0,
                    use_cross_val: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Train a Radial Basis Function (RBF) Surrogate Model

    Args:
    - X_train (np.ndarray): Training feature data (num_samples, num_features)
    - Y_train (np.ndarray): Training target data (num_samples, num_outputs)
    - num_centers (int): Number of RBF centers. Key hyperparameter
    - gamma (Union[float, str]): RBF kernel coefficient. 'scale' uses a heuristic
    - alpha (float): Regularization strength (lambda) for the final Ridge regression
    - use_cross_val (bool): Whether to perform cross-validation to assess model stability.
    - cv_folds (int): Number of folds for K-fold cross-validation

    Returns:
    - Dict[str, Any]: A dictionary containing the trained model, RBF transformer, and cross-validation results
    """

    # Feature Engineering: Determine centers, bandwidth, and transform features
    rbf_transformer = RBF_Transformer(num_centers=num_centers, gamma=gamma, random_state=42)
    X_rbf = rbf_transformer.fit_transform(X_train)  # (num_samples, num_centers)

    # Model Selection: Choose the regression algorithm (Ridge for regularization)
    if alpha > 0.0:
        # Ridge regression
        model = Ridge(alpha=alpha, solver='auto', random_state=42)
        model_name = 'Ridge RBF Model'
    else:
        # Standard linear regression
        model = LinearRegression()
        model_name = 'Standard RBF Model'

    # Model Training: Fit the model to the RBF (Kernel) features
    print(f'# Training {model_name} with num_centers={num_centers}, gamma={rbf_transformer.gamma:.4f}, alpha={alpha}...')
    model.fit(X_rbf, Y_train)
    print(f'# {model_name} training completed')

    # Cross-validation: Assess generalization performance
    cv_results = {}
    if use_cross_val:
        cv_scores = cross_val_score(estimator=model, X=X_rbf, y=Y_train, scoring='r2',
                                    cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42), n_jobs=-1)
        
        cv_results = {
            'cv_folds':cv_folds,
            'cv_mean_scores': np.mean(cv_scores),
            'cv_std_scores': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }

    # Return the trained model
    return {
        'model_name': model_name,
        'model': model,
        'rbf_transformer': rbf_transformer,
        'num_centers': num_centers,
        'gamma': rbf_transformer.gamma,
        'alpha': alpha,
        'cv_results': cv_results
    }


def test_rbf_model(trained_model_dict: Dict[str, Any],
                   X_test: np.ndarray, Y_test: Optional[np.ndarray] = None
                   ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    Evaluate the trained Radial Basis Function (RBF) Surrogate Model on test data

    Args:
    - trained_model_dict (Dict[str, Any]): Dictionary returned by train_rbf_model
    - X_test (np.ndarray): Test feature data (num_samples, num_features)
    - Y_test (np.ndarray, Optional): Test target data (num_samples, num_outputs)

    Returns:
    - Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    - If Y_test is available: Returns a Tuple: (Y_pred, metrics)
    - If Y_test is None: Returns only Y_pred

    - Y_pred (np.ndarray): Predicted target values
    - metrics (Dict[str, float]): Dictionary of evaluation metrics
    """

    # Retrieve the trained components
    model_name = trained_model_dict['model_name']
    model: RBFModel = trained_model_dict['model']
    rbf_transformer: RBF_Transformer = trained_model_dict['rbf_transformer']

    # Feature Transformer: Apply the RBF transformation to the test data
    X_test_rbf = rbf_transformer.transform(X_test)

    # Prediction: Generate predictions on the transformed test features
    print(f'# Predicting {model_name}...')
    Y_pred = model.predict(X_test_rbf)
    print(f'# {model_name} prediction completed')

    # Return predictions if Y_test is not available (Inference Mode)
    if Y_test is None:
        return Y_pred

    # Metrics Calculation: Evaluate model performance (Evaluation Mode)
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse
    }

    # Return predictions and performance metrics
    return Y_pred, metrics


# ======================================================================
# Example Usage
# ======================================================================
if __name__ == '__main__':
    # Simulate data
    np.random.seed(42)
    N = 1000
    X = np.random.rand(N, 2) * 10

    # Branin function
    y1 = 1.0 * (X[:, 1] - 5.1 / (4.0 * np.pi**2) * X[:, 0]**2 + 5.0 / np.pi * X[:, 0] - 6.0)**2 + \
    10.0 * (1 - 1.0 / (8.0 * np.pi)) * np.cos(X[:, 0]) + 10.0

    # Simple interaction
    y2 = X[:, 0] * X[:, 1] + np.sin(X[:, 0]) * 10

    Y = np.stack([y1, y2], axis=1)

    # Split data into training and testing sets (8/2 split)
    split_idx = int(0.8 * N)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Train model
    model = train_rbf_model(X_train, Y_train, num_centers=150, gamma='scale', alpha=0.0)

    # Show cross-validation results
    print(f'# Cross-validation Mean R2: {model['cv_results']['cv_mean_scores']:.9f}')

    # Test model
    Y_pred, test_metrics = test_rbf_model(model, X_test, Y_test)

    # Show testing results
    print(f'# Testing R2: {test_metrics['r2']:.9f}')
