import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Set LOKY_MAX_CPU_COUNT to silence the core detection warning
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

### load data
X_train = np.load("Data/X_train.npy") 
y_train = np.load("Data/y_train.npy") # 6k male (m), 2k female (f)
X_test = np.load("Data/X_test.npy")
y_test = np.load("Data/y_test.npy") # 2k male (m), 2k female (f)

# 2D
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)

# Plot masculine (m) data points
plt.scatter(X_train_tsne[y_train == 'm', 0], X_train_tsne[y_train == 'm', 1], c='blue', label='Masculine', alpha=0.6)

# Plot feminine (f) data points
plt.scatter(X_train_tsne[y_train == 'f', 0], X_train_tsne[y_train == 'f', 1], c='red', label='Feminine', alpha=0.6)

plt.title('t-SNE visualization of Masculine and Feminine data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.legend(loc='best')

plt.show()

# 3D
### feature normalization
scaler = StandardScaler()
# scaler must be fitted on training data only
X_scaled = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA to reduce dimensions before t-SNE
pca = PCA(n_components=50)  # Reduce to 50 dimensions before t-SNE
X_pca = pca.fit_transform(X_scaled)

# Initialize t-SNE with different parameters
tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)

# Fit and transform the data
X_tsne = tsne.fit_transform(X_pca)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot masculine (m) data points
ax.scatter(X_tsne[y_train == 'm', 0], X_tsne[y_train == 'm', 1], X_tsne[y_train == 'm', 2], c='blue', label='Masculine', alpha=0.6)

# Plot feminine (f) data points
ax.scatter(X_tsne[y_train == 'f', 0], X_tsne[y_train == 'f', 1], X_tsne[y_train == 'f', 2], c='red', label='Feminine', alpha=0.6)

# Add title and labels
ax.set_title('3D t-SNE visualization of Masculine and Feminine data')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')

plt.legend(loc='best')

plt.show()