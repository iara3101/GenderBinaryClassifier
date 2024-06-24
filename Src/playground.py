import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load data
X_train = np.load("Data/X_train.npy")
y_train = np.load("Data/y_train.npy")  # 6k male (m), 2k female (f)
X_test = np.load("Data/X_test.npy")
y_test = np.load("Data/y_test.npy")  # 2k male (m), 2k female (f)


print(X_train)