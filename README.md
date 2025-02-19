This Model aims at predicting Air quality as 'Good', 'Hazardous', 'Moderate', 'Poor' bases on several parameters.

Following are the independent variables that are used to predict the air quality:
'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO','Proximity_to_Industrial_Areas', Population_Density'

Project structure and flow:
- data_ingestion.py file is used to read the data from csv file store in the artifact directory after splitiing into train and test data.
- data_transaformation.py file is used to transform the columns wherever required. Transformation pipeline is created using ColumnTransformer.
- model_trainer.py is used to pick the best model, train the model and store the pickle file.
- app.py in the entry point for the application.

For CI/CD - git repository and AWS ECR are used.
*Note: AWS app runner URI is no more live for use.

Steps for CICD:
- Create dockerfile.
- Create yaml file with steps to CI and CD
- On AWS, create a ECR repository, IAM user
- Add variables Settlings/secrets and variables
- On AWS, create Apprunner service and link it with ECR repo.
- After successfull deployment of apprunner service, AWS will provide URI for the website.
