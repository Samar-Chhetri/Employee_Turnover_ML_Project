import os, sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, satisfaction_level:float, last_evaluation:float, number_project:int, average_montly_hours:int, time_spend_company:int,
                 Work_accident:int, promotion_last_5years:int, department:str, salary:str):
        
        self.satisfaction_level = satisfaction_level
        self.last_evaluation = last_evaluation
        self.number_project = number_project
        self.average_montly_hours = average_montly_hours
        self.time_spend_company = time_spend_company
        self.Work_accident = Work_accident
        self.promotion_last_5years = promotion_last_5years
        self.department = department
        self.salary = salary

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "satisfaction_level": [self.satisfaction_level],
                "last_evaluation": [self.last_evaluation],
                "number_project": [self.number_project],
                "average_montly_hours": [self.average_montly_hours],
                "time_spend_company": [self.time_spend_company],
                "Work_accident": [self.Work_accident],
                "promotion_last_5years": [self.promotion_last_5years],
                "department": [self.department],
                "salary": [self.salary]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)







