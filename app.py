
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
    # app.run(host="0.0.0.0",port=8080)        
    app.run(host='0.0.0.0', port=8080)  
    predict_pipeline=PredictPipeline()
    #results=predict_pipeline.predict([[26.5,70.7,6.9,16,21.9,5.6,1.01,12.7,303]])  # Good
    #results=predict_pipeline.predict([[39.4,96.6,14.6,35.5,42.9,17.9,1.82,3.1,674]])  #Hazardous
    #results=predict_pipeline.predict([[0.28982301, 0.50162866, 0.09954186, 0.13068399, 0.23699422,0.23287671, 0.21498371, 0.11587983, 0.32509753]])
    #results=predict_pipeline.predict([[0.44026549, 0.04234528, 0.06122449, 0.08024455, 0.23699422,0.28571429, 0.13355049, 0.37339056, 0.19635891]])
    #results=predict_pipeline.predict([[0.42256637, 0.36482085, 0.15951687, 0.18494459, 0.50674374,0.32876712, 0.19218241, 0.15450644, 0.22106632]])
    

    #print(results)
        
 