
import os
import sys
from dataclasses import dataclass
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from src.exception import CustomException
from src.logger import logging
from sklearn import metrics

from src.utils import save_object,evaluate_models,perf_eval_mul

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            #logging.info(train_array,test_array)
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
          
            
            param = [1,2]
            clf = SVC(random_state=0, probability=True)
            clf1 = GridSearchCV(estimator= clf, param_grid={'C': param }, cv=10).fit(X_train,y_train)
            yp = cross_val_predict(clf1.best_estimator_, X_train, y_train, cv=10, method ='predict_proba')
            model_report=perf_eval_mul(y_train,yp)
            
            
            ## To get best model score from dict
            clf2 = clf1.best_estimator_
            clf2.fit(X_train, y_train)
          
            best_model = clf2

 

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            ACC = metrics.balanced_accuracy_score(y_test, predicted)
            
            return ACC
            



            
        except Exception as e:
            raise CustomException(e,sys)
