import os, sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours','time_spend_company',
                             'Work_accident','promotion_last_5years']
            categorical_nominal_column = ['department']
            categorical_ordinal_column = ['salary']

            cat_nominal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
                ('ss', StandardScaler(with_mean=False))
            ])

            cat_ordinal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('oe', OrdinalEncoder(categories=[['low', 'medium', 'high']])),
                ('ss', StandardScaler(with_mean=False))
            ])

            num_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='median')),
                ('ss', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical ordinal column, {categorical_ordinal_column}")
            logging.info(f"Categorical nominal column, {categorical_nominal_column}")
            logging.info(f"Numerical columns, {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('cat_nominal_pipeline', cat_nominal_pipeline, categorical_nominal_column),
                ('cat_ordinal_pipeline', cat_ordinal_pipeline, categorical_ordinal_column),
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            train_df.rename(columns={'sales': 'department'}, inplace=True)
            test_df.rename(columns={'sales': 'department'}, inplace=True)

            target_column_name = ['left']

            input_feature_train_df = train_df.drop(columns=target_column_name)
            target_feature_train_df = train_df['left']


            input_feature_test_df = test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df['left']

            logging.info("Applying preprocessor object on train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e, sys)
        











