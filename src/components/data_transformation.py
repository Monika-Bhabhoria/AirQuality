
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns =['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
       'Proximity_to_Industrial_Areas', 'Population_Density']
            categorical_columns = []

            num_pipeline= Pipeline(
                steps=[
                #("imputer",SimpleImputer(strategy="median")),
                ("scaler",MinMaxScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",MinMaxScaler())
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,[0,1,2,3,4,5,6,7,8])
                #("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            #print(train_df.columns)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="AirQuality"
            numerical_columns = ['Temperature','Humidity','PM2.5','PM10','NO2','SO2','CO','Proximity_to_Industrial_Areas','Population_Density','AirQuality']
            col = train_df.columns.tolist().index('AirQuality')
            i1 = train_df.iloc[:,col] == 'Good'
            i2 = train_df.iloc[:,col] == 'Hazardous'
            i3 = train_df.iloc[:,col] == 'Moderate'
            i4 = train_df.iloc[:,col] == 'Poor' 

            train_df.iloc[i1,col] = 0
            train_df.iloc[i2,col] = 1
            train_df.iloc[i3,col] = 2
            train_df.iloc[i4,col] = 3
            train_df['AirQuality'] = train_df['AirQuality'].astype('int')

            col = test_df.columns.tolist().index('AirQuality')
            i1 = test_df.iloc[:,col] == 'Good'
            i2 = test_df.iloc[:,col] == 'Hazardous'
            i3 = test_df.iloc[:,col] == 'Moderate'
            i4 = test_df.iloc[:,col] == 'Poor' 

            test_df.iloc[i1,col] = 0
            test_df.iloc[i2,col] = 1
            test_df.iloc[i3,col] = 2
            test_df.iloc[i4,col] = 3
            test_df['AirQuality'] = test_df['AirQuality'].astype('int')

            
            input_feature_train_df=train_df.drop('AirQuality',axis=1)
            #print(input_feature_train_df.columns)
            target_feature_train_df=train_df['AirQuality']
            
            
            input_feature_test_df=test_df.drop('AirQuality',axis=1)
            target_feature_test_df=test_df['AirQuality']
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df.values)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df.values)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
   
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation("//artifacts/train.csv","//artifacts/test.csv")

 
