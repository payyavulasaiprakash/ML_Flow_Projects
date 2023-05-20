# importing required values
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, pickle
import category_encoders as ce
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import  RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import mlflow

csv_file_name = 'train_v9rqX0R.csv'

def prepare_dataset(csv_file_name=csv_file_name):
    # read the train data
    dataset = pd.read_csv(csv_file_name)

    dataset.Item_Weight.fillna(dataset.Item_Weight.median(),inplace=True)

    dataset.Outlet_Size.fillna(dataset.Outlet_Size.mode()[0],inplace=True)
    print('columns',dataset.columns) 
    OHE = ce.OneHotEncoder(cols=['Item_Fat_Content',
                             'Item_Type',
                             'Outlet_Identifier',
                             'Outlet_Size',
                             'Outlet_Location_Type',
                             'Outlet_Type'],use_cat_names=True)
# encode the categorical variables
    dataset = OHE.fit_transform(dataset)
    dataset_X = dataset.drop(columns=['Item_Identifier','Item_Outlet_Sales'])
    dataset_Y = dataset['Item_Outlet_Sales']
    

    train_x, test_x, train_y, test_y = train_test_split(dataset_X, dataset_Y,test_size=0.25,random_state=0)

    scaler = StandardScaler()
    # fit with the Item_MRP
    scaler.fit(np.array(train_x.Item_MRP).reshape(-1,1))
    # transform the data
    train_x.Item_MRP = scaler.transform(np.array(train_x.Item_MRP).reshape(-1,1))
    test_x.Item_MRP = scaler.transform(np.array(test_x.Item_MRP).reshape(-1,1))

    with open('one_hot_encoding.pkl','wb') as file:
        pickle.dump(OHE,file)

    with open('StandardScaler.pkl','wb') as file:
        pickle.dump(scaler,file)
    return train_x, test_x, train_y, test_y 


def fit(train_x,train_y,estimator):
    estimator = estimator
    estimator.fit(train_x, train_y)
    return estimator

def test(data,estimator,standard_scaler,scaling = False,):
    if scaling:
        data.Item_MRP = standard_scaler.transform(np.array(data.Item_MRP).reshape(-1,1))
        predict_test  = estimator.predict(data)
    else:
        predict_test  = estimator.predict(data)
    return predict_test


train_x, test_x, train_y, test_y  =  prepare_dataset(csv_file_name=csv_file_name)
with open('StandardScaler.pkl','rb') as file:
    standard_scaler = pickle.load(file)

EXPERIMENT_NAME = "mlflow-Ridge-1"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

for alpha in [0.5,0.05,0.005,0.0005,0.1,0.001,0.25,1,100,10,0.00001,0.0001,0.0025,0.025,0.02]:

    model_ridge = Ridge(alpha=alpha,normalize=True)
    model_ridge = fit(train_x,train_y,model_ridge)

    model_ridge_predictions_train = test(train_x,model_ridge,standard_scaler, False)
    model_ridge_predictions_test = test(test_x,model_ridge,standard_scaler,False)

    # Root Mean Squared Error on train and test date

    model_ridge_train_rmse = mean_squared_error(train_y, model_ridge_predictions_train)**(0.5)
    model_ridge_test_rmse = mean_squared_error(test_y, model_ridge_predictions_test)**(0.5)

    print(alpha,model_ridge_train_rmse, model_ridge_test_rmse)

    RUN_NAME = f'alpha_{alpha}'

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
        RUN_ID = run.info.run_id

        # Track parameters
        mlflow.log_param("alpha", alpha)

        mlflow.log_metric("RR_RSME_train_data_error", model_ridge_train_rmse)

        # Track metrics
        mlflow.log_metric("RR_RSME_test_data_error", model_ridge_test_rmse)

        # Track model
        mlflow.sklearn.log_model(model_ridge, "Ridge_Regression")


    # with open('ridge_model.pkl','wb') as file:
    #         pickle.dump(model_LR,file)


EXPERIMENT_NAME = "mlflow-Random_Forest-1"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

for max_depth in [1,3,6,7,9,10,20,30,5,4,2,8,15,25]:

    # create an object of the RandomForestRegressor
    model_RFR = RandomForestRegressor(max_depth=max_depth)

    # fit the model with the training data
    model_RFR.fit(train_x, train_y)

    # predict the target on train and test data
    predict_train = model_RFR.predict(train_x)
    predict_test = model_RFR.predict(test_x)

    RUN_NAME = f'max_depth_{max_depth}'

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
        RUN_ID = run.info.run_id

        # Track parameters
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("RFR_RSME_train_data_error", model_ridge_train_rmse)

        # Track metrics
        mlflow.log_metric("RFR_RSME_test_data_error", model_ridge_test_rmse)

        # Track model
        mlflow.sklearn.log_model(model_ridge, "Random_Forest_Regression")



