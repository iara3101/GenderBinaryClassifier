import numpy as np
from sklearn.manifold import TSNE

### load data
X_train = np.load("Data/X_train.npy") 
y_train = np.load("Data/y_train.npy") # 6k male (m), 2k female (f)
X_test = np.load("Data/X_test.npy")
y_test = np.load("Data/y_test.npy") # 2k male (m), 2k female (f)

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_train, y_train)
X_embedded.shape(4, 2)