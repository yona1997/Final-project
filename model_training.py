import pandas as pd
import numpy as np
import re
import datetime
from datetime import timedelta
import string

from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder,MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from madlan_data_prep import prepare_data
from sklearn.model_selection import GridSearchCV


y = model_data['price']
X = model_data.drop('price', axis=1)

excel_file = 'output_all_students_Train_v10.xlsx'
model_data = pd.read_excel(excel_file)
model_data = prepare_data(model_data)

num_featur = ['room_number', 'Area']
ctg_featur = ['City', 'type']

# Initialize the encoder
encoder = OneHotEncoder(handle_unknown='ignore')

scaler = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', encoder, ctg_featur),
        ('scaler', scaler, num_featur)
    ], remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaled_X_train = preprocessor.fit_transform(X_train)
scaled_X_test = preprocessor.transform(X_test) 


param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.2, 0.4, 0.6, 0.8],
}

grid_search = GridSearchCV(ElasticNet(random_state=42), param_grid, cv=10, scoring='r2')
grid_search.fit(scaled_X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("most hyper parameter:", best_params)
print("most CV Score:", best_score)


model = ElasticNet(alpha=1, l1_ratio=0.6, random_state=42)
model.fit(scaled_X_train, y_train)

y_pred = cross_val_predict(model, scaled_X_test, y_test, cv=10)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

cv_scores = cross_val_score(model, scaled_X_train, y_train, cv=10, scoring='r2')
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
