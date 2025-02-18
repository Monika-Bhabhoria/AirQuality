
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
            # col = train_array.columns.tolist().index('AirQuality')
            # i1 = train_array.iloc[:,col] == 'Good'
            # i2 = train_array.iloc[:,col] == 'Hazardous'
            # i3 = train_array.iloc[:,col] == 'Moderate'
            # i4 = train_array.iloc[:,col] == 'Poor' 

            # train_array.iloc[i1,col] = 0
            # train_array.iloc[i2,col] = 1
            # train_array.iloc[i3,col] = 2
            # train_array.iloc[i4,col] = 3
            # train_array['AirQuality'] = train_array['AirQuality'].astype('int')

            # col = test_array.columns.tolist().index('AirQuality')
            # i1 = test_array.iloc[:,col] == 'Good'
            # i2 = test_array.iloc[:,col] == 'Hazardous'
            # i3 = test_array.iloc[:,col] == 'Moderate'
            # i4 = test_array.iloc[:,col] == 'Poor' 

            # test_array.iloc[i1,col] = 0
            # test_array.iloc[i2,col] = 1
            # test_array.iloc[i3,col] = 2
            # test_array.iloc[i4,col] = 3
            # test_array['AirQuality'] = test_array['AirQuality'].astype('int')


            # X_train=train_array.drop('AirQuality',axis=1)
            # y_train=train_array['AirQuality']
            # X_test=test_array.drop('AirQuality',axis=1)
            # y_test=test_array['AirQuality']

            # from sklearn.preprocessing import MinMaxScaler
            # scl = MinMaxScaler().fit(X_train)
            # X_train = scl.transform(X_train)
            # X_test = scl.transform(X_test)
            
            param = [1,2]
            clf = SVC(random_state=0, probability=True)
            clf1 = GridSearchCV(estimator= clf, param_grid={'C': param }, cv=10).fit(X_train,y_train)
            yp = cross_val_predict(clf1.best_estimator_, X_train, y_train, cv=10, method ='predict_proba')
            model_report=perf_eval_mul(y_train,yp)
            
            
            ## To get best model score from dict
            clf2 = clf1.best_estimator_
            clf2.fit(X_train, y_train)
            #ytp = clf2.predict_proba(X_test)

            ## To get best model name from dict

            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            best_model = clf2

            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            ACC = metrics.balanced_accuracy_score(y_test, predicted)
            
            return ACC
            



            
        except Exception as e:
            raise CustomException(e,sys)
