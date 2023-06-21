import os, sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")

            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boost": GradientBoostingClassifier(),
                "Support Vector Classifier": SVC(),
                "K-nearest Classifier": KNeighborsClassifier()
            }

            params ={
                "Logistic Regression": {},

                "Decision Tree": {
                    'max_depth': [3,4,5,6,7],
                    'criterion': ['gini','entropy']
                },

                "Random Forest": {
                    'max_depth': [3,4,5,6,7,9],
                    'criterion': ['gini','entropy']
                },

                "AdaBoost": {
                    'n_estimators': [30,50,70,100],
                    'learning_rate': [0.1, 0.01, 0.5, 0.05]
                },

                "Gradient Boost": {
                    'criterion': ['friedman_mse', 'squared_error']
                },

                "Support Vector Classifier": {},

                "K-nearest Classifier": {
                    'n_neighbors': [3,5,7,9]
                }

            }



            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # Get best model name
            best_model_name = list(model_report.keys()) [list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.7:
                raise CustomException("No best model found")
            logging.info("Best model found on both train and test set")

            save_object(
                file_path =self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            f1_sco = f1_score(y_test, predicted)
            return f1_sco
        
            
        except Exception as e:
            raise CustomException(e, sys)