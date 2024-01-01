import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = list(params.values())[i]
            # logging.info("{}".format(para))
            grid_search = GridSearchCV(model, para, cv=5, scoring='neg_mean_squared_error')
            logging.info("{}".format(grid_search))
            grid_search.fit(x_train, y_train)
            model.set_params(**grid_search.best_params_)
            model.fit(x_train,y_train)
            #model.fit(x_train,y_train)
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)

            train_model_score = r2_score(y_train,y_pred_train)
            test_model_score = r2_score(y_test,y_pred_test)

            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e,sys)