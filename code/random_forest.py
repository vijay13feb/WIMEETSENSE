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
from sklearn.ensemble import RandomForestClassifier
folder = sys.argv[1]
#loading training testing dataset 

feat_train_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_train.csv'))
label_train_amp=pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_train.csv'))
feat_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_test.csv'))
label_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_test.csv'))

# training the model 
random_forest = RandomForestClassifier(n_estimators=1000, max_depth=4)
random_forest_ = random_forest.fit(feat_train_amp, label_train_amp)


predict_train = random_forest_.predict(feat_train_amp)
predict_test = random_forest_.predict(feat_test_amp)
# evaluation
print("training", classification_report(predict_train, label_train_amp))
print("test", classification_report(predict_test, label_test_amp))
#optional
# joblib.dump(random_forest_, os.path.abspath(f'./model/rf{S1}.sav'))



# 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, val_index in kf.split(feat_train_amp):
    X_train, X_val = feat_train_amp.iloc[train_index], feat_train_amp.iloc[val_index]
    y_train, y_val = label_train_amp[train_index], label_train_amp[val_index]
    
    # Train the model
    random_forest.fit(X_train, y_train)
    
    # Predict on the validation set
    y_val_pred = random_forest.predict(X_val)
    
    # Calculate accuracy and F1 score
    accuracy_scores.append(accuracy_score(y_val, y_val_pred))
    f1_scores.append(f1_score(y_val, y_val_pred, average='weighted'))

# Compute the average accuracy and F1 score
mean_accuracy = np.mean(accuracy_scores)
mean_f1_score = np.mean(f1_scores)


print("Mean 10- cross-validation accuracy score:", mean_accuracy)

print("Mean 10-cross-validation F1 score:", mean_f1_score)