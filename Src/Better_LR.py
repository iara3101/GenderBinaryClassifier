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

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('oversample', SMOTE(sampling_strategy='auto', random_state=42)),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Define hyperparameters for GridSearch
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__solver': ['liblinear', 'saga']
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate on validation set
y_val_pred = best_model.predict(X_val)
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
print("Validation ROC AUC Score:", roc_auc_score(y_val, y_val_pred))

# Evaluate on the test set
y_test_pred = best_model.predict(X_test)
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
print("Test ROC AUC Score:", roc_auc_score(y_test, y_test_pred))

# Accuracy per class on the test set
idx_m = y_test == "m"
idx_f = y_test == "f"

acc_m = np.mean(y_test_pred[idx_m] == y_test[idx_m])
acc_f = np.mean(y_test_pred[idx_f] == y_test[idx_f])
print("Test Accuracy M: {}".format(acc_m))
print("Test Accuracy F: {}".format(acc_f))
