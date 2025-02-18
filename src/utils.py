
import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def perf_eval_mul(y_test, y_prob):  
    yp = np.argmax(y_prob, axis=1)
    ACC = metrics.balanced_accuracy_score(y_test, yp)
    REC = metrics.recall_score(y_test, yp, average='macro')
    PRE = metrics.precision_score(y_test, yp, average='macro')
    MCC = metrics.matthews_corrcoef(y_test, yp)
    F1 = metrics.f1_score(y_test, yp, average='macro')
    y_onehot_test = LabelBinarizer().fit_transform(y_test)
    AUC = metrics.roc_auc_score(y_onehot_test, y_prob, average='macro', multi_class='ovr')
    #print("ACC: ",ACC," MCC: ",MCC," AUC:",AUC, " PRE: ",PRE," REC: ",REC," F1: ",F1)
    return [ACC, MCC, AUC, PRE, REC, F1]
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
