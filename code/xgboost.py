#python xgboost.py S1
import pandas as pd
import pandas as pd 
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import os
import sys
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from xgboost import XGBClassifier


folder = sys.argv[1]
#loading training testing dataset 

feat_train_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_train.csv'))
label_train_amp=pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_train.csv'))
feat_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_test.csv'))
label_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_test.csv'))

# training the model 
label_train_amp.headlabel.replace(('Forward', 'Looking Down', 'Looking Up', 'Looking Left', 'Looking Right','Nodding', 'Shaking' ), (0, 1, 2,3, 4,5,6), inplace=True)
label_test_amp.headlabel.replace(('Forward', 'Looking Down', 'Looking Up', 'Looking Left', 'Looking Right','Nodding', 'Shaking' ), (0, 1, 2,3, 4,5,6), inplace=True)
xg_amp = XGBClassifier(learning_rate = 0.1, n_estimators= 1000, max_depths=5, min_child_weight=1, gamma = 0,subsample=0.5, colsample_bytree=0.8, objective='multi:softprob', scale_pos_weight=19.44, seed=27, tree_method='gpu_hist')
xg_amp.fit(feat_train_amp, label_train_amp)

predict_train = xg_amp.predict(feat_train_amp)
predict_test = xg_amp.predict(feat_test_amp)
# compute classification accuracy
print("training", classification_report(predict_train, label_train_amp))
print("test", classification_report(predict_test, label_test_amp))
#optional
# joblib.dump(xg_amp, os.path.abspath(f'./model/xg{S1}.sav'))


# 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, val_index in kf.split(feat_train_amp):
    X_train, X_val = feat_train_amp.iloc[train_index], feat_train_amp.iloc[val_index]
    y_train, y_val = label_train_amp[train_index], label_train_amp[val_index]
    
    # Train the model
    xg_amp.fit(X_train, y_train)
    
    # Predict on the validation set
    y_val_pred = xg_amp.predict(X_val)
    
    # Calculate accuracy and F1 score
    accuracy_scores.append(accuracy_score(y_val, y_val_pred))
    f1_scores.append(f1_score(y_val, y_val_pred, average='weighted'))

# Compute the average accuracy and F1 score
mean_accuracy = np.mean(accuracy_scores)
mean_f1_score = np.mean(f1_scores)


print("Mean 10- cross-validation accuracy score:", mean_accuracy)

print("Mean 10-cross-validation F1 score:", mean_f1_score)

