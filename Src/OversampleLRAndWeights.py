import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import os

# Set LOKY_MAX_CPU_COUNT to silence the core detection warning
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

### load data
X_train = np.load("Data/X_train.npy") 
y_train = np.load("Data/y_train.npy") # 6k male (m), 2k female (f)
X_test = np.load("Data/X_test.npy")
y_test = np.load("Data/y_test.npy") # 2k male (m), 2k female (f)

### feature normalization
scaler = StandardScaler()
# scaler must be fitted on training data only
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Oversample the minority class using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

### Define model with class weights
class_weights = {'m': 1, 'f': len(y_train[y_train == 'm']) / len(y_train[y_train == 'f'])}
model = LogisticRegression(max_iter=1000, class_weight=class_weights, solver='liblinear')

### training
model.fit(X_train_res, y_train_res)

### evaluation

# accuracy (the test data is balanced)
acc = model.score(X_test, y_test)
print("Acc: {}".format(acc))
 
# accuracy per class
y_pred = model.predict(X_test)

idx_m = y_test=="m"
idx_f = y_test=="f"

acc_m = model.score(X_test[idx_m], y_test[idx_m])
acc_f = model.score(X_test[idx_f], y_test[idx_f])
print("Acc M: {}".format(acc_m))
print("Acc F: {}".format(acc_f))

# Detailed classification report
print(classification_report(y_test, y_pred))