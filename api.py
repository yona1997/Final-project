from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime
import re

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("property_prediction.pkl", "rb"))
X = pickle.load(open("X_cat.pkl","rb")).astype(str)
x_cat =X[:, [0, 1, 13, 19, 24]]

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the property data from the form
    city = request.form['city']
    property_type = request.form['type']
    room_number = float(request.form['room_number'])
    area = float(request.form['area'])
    has_elevator = int(request.form['hasElevator'])
    has_parking = int(request.form['hasParking'])
    has_bars = int(request.form['hasBars'])
    has_storage = int(request.form['hasStorage'])
    has_air_condition = int(request.form['hasAirCondition'])
    has_balcony = int(request.form['hasBalcony'])
    has_mamad = int(request.form['hasMamad'])
    handicap_friendly = int(request.form['handicapFriendly'])
    published_days = float(request.form['publishedDays'])
    floor_out_of = float(request.form['floor_out_of'])
    floor = float(request.form['floor'])
    
    entrance_date = request.form['entrance_date']
    
    new_row = np.array([city, property_type, area, has_elevator, has_parking, has_bars,
                    has_storage, has_air_condition, has_balcony, has_mamad,
                    handicap_friendly, published_days, floor_out_of, floor])


    # Insert the new row at the beginning
    new_array = np.insert(x_cat, 0, new_row, axis=0)
    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit and transform the categorical columns in the training data
    X_encoded = encoder.fit_transform(x_cat)

    # Convert the encoded data to dense arrays
    X_encoded = X_encoded.toarray()    

    
    # Make the prediction using the loaded model
    predicted_price = model.predict(property_data)
    
    # Render the predicted price in the HTML template
    return render_template('index.html', predicted_price=predicted_price[0])

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)