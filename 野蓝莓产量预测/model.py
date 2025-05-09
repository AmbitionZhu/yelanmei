import joblib 
import pandas as pd
import numpy as np
import os

# load the model file
curr_path = os.path.dirname(os.path.realpath(__file__))
xgb_model = joblib.load(curr_path + "/model/wbb_xgb_model2.joblib")

# function to predict the yield
def predict_yield(attributes: np.ndarray):
    """ Returns Blueberry Yield value"""
    # print(attributes.shape) # (1,8)

    pred = xgb_model.predict(attributes)
    print("Yield predicted")

    return pred[0]