# kNN implementation

from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import cdist
class kNN(BaseEstimator, ClassifierMixin):
  def __init__(self, n_neighbors:int = 3):
    self.n_neighbors = n_neighbors

  def fit(self, X, y):
    self.X_train = np.copy(X)
    self.y_train = np.copy(y)
    return self

  def predict(self, X):
    distances = cdist(np.copy(X), self.X_train)
    indices = (np.argpartition(distances, self.n_neighbors-1))[:,:self.n_neighbors]
    labels = self.y_train.flatten()[indices]
    predictions = np.sign(np.sum(labels, axis=1))
    return predictions
