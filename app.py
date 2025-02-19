
from flask import Flask,request,render_template
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.logger import logging

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        print("pred_df")
        return render_template('home.html')
    else:
        print("pred_df")
        data=CustomData(
            Temperature=float(request.form.get('Temperature')),
            Humidity=float(request.form.get('Humidity')),
            PM25=float(request.form.get('PM25')),
            PM10=float(request.form.get('PM10')),
            NO2=float(request.form.get('NO2')),
            SO2=float(request.form.get('SO2')),
            CO=float(request.form.get('CO')),
            Proximity_to_Industrial_Areas=float(request.form.get('Proximity_to_Industrial_Areas')),
            Population_Density=request.form.get('Population_Density'))
        
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
if __name__=="__main__":
        
    app.run(host='0.0.0.0', port=8080)  
    
    predict_pipeline=PredictPipeline()
  
 