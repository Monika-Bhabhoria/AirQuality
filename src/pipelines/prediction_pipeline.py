
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Temperature: float,
        Humidity: float,
        PM25 : float,
        PM10: float,
        NO2: float,
        SO2: float,
        CO: float,
        Proximity_to_Industrial_Areas:float,
        Population_Density:int):
        self.Temperature = Temperature
        self.Humidity = Humidity
        self.PM25 = PM25
        self.PM10 = PM10
        self.NO2 = NO2
        self.SO2 = SO2
        self.CO = CO
        self.Proximity_to_Industrial_Areas = Proximity_to_Industrial_Areas
        self.Population_Density = Population_Density

    def get_data_as_data_frame(self):
        try:
            
            logging.info("Converting data to data frame")
            custom_data_input_dict = {
                "Temperature": [self.Temperature],
                "Humidity": [self.Humidity],
                "PM25": [self.PM25],
                "PM10": [self.PM10],
                "NO2": [self.NO2],
                "SO2": [self.SO2],
                "CO": [self.CO],
                "Proximity_to_Industrial_Areas": [self.Proximity_to_Industrial_Areas],
                "Population_Density": [self.Population_Density]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
