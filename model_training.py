import pandas as pd
import numpy as np
import re
import datetime
from datetime import timedelta
import string
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from madlan_data_prep import prepare_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder,MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.linear_model import ElasticNet, ElasticNetCV

excel_file = 'output_all_students_Train_v10.xlsx'
data = pd.read_excel(excel_file)
data = prepare_data(data)

y = data['price']
X = data.drop('price', axis=1)
num_co = ['room_number', 'Area']
ctg_co = ['City', 'type']

# Initialize the encoder
encoder = OneHotEncoder(handle_unknown='ignore')
scaler = MinMaxScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', encoder, ctg_co),
        ('scaler', scaler, num_co)
    ], remainder='passthrough'
)

#divise the dataset train / test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc_X_train = preprocessor.fit_transform(X_train)
sc_X_test = preprocessor.transform(X_test) 

param = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.2, 0.4, 0.6, 0.8],
}

param_search = GridSearchCV(ElasticNet(random_state=42), param, cv=10, scoring='r2')
param_search.fit(sc_X_train, y_train)

best_params = param_search.best_params_
best_score = param_search.best_score_

#we take alpha = 1 and l1 ration = 0.6

model = ElasticNet(alpha=1, l1_ratio=0.6, random_state=42)
model.fit(sc_X_train, y_train)

y_pred = cross_val_predict(model, sc_X_test, y_test, cv=10)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

cv_scores = cross_val_score(model, sc_X_train, y_train, cv=10, scoring='r2')
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()

print("Mean CV", mean_cv_score)
print("Std CV", std_cv_score)
print("R2", r2)
print("RMSE" + str(np.sqrt(mse))) 

# ----------------------------------------------------------------------------

import pickle
pickle.dump(model, open("trained_model.pkl", "wb"))
pickle.dump(preprocessor, open("preprocessor.pkl", 'wb'))
