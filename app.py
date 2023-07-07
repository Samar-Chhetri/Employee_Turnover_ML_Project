from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/empturnover', methods =['GET','POST'])
def predict_turnover():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            satisfaction_level = float(request.form.get('satisfaction_level')),
            last_evaluation = float(request.form.get('last_evaluation')),
            number_project = int(request.form.get('number_project')),
            average_montly_hours = int(request.form.get('average_montly_hours')),
            time_spend_company = int(request.form.get('time_spend_company')),
            Work_accident = int(request.form.get('Work_accident')),
            promotion_last_5years = int(request.form.get('promotion_last_5years')),
            department = request.form.get('department'),
            salary = request.form.get('salary')
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results= results[0])


if __name__ =="__main__":
    app.run(host="0.0.0.0")
