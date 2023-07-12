import pandas as pd
import numpy as np
import re
import datetime
from datetime import timedelta
import string

path = "output_all_students_Train_v10.xlsx"
data = pd.read_excel(path)
def price(data):
    data['price'] = data['price'].astype(str).apply(lambda x: re.findall('[0-9]+', x))
    data['price'] = data['price'].apply(lambda x: ''.join(x))
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data = data.dropna(subset=['price'])

def room_number(data):
    data['room_number'] = data['room_number'].apply(lambda x: str(x).replace('חד׳', ''))
    data['room_number'] = pd.to_numeric(data['room_number'], errors='coerce')
    data['room_number'] = data['room_number'].replace('', np.nan)
    data['room_number'] = data['room_number'].astype(float)
    data.loc[data['room_number'] > 10, 'room_number'] = np.nan
    data.dropna(subset=['room_number'], inplace=True)

def area(data):
    data['Area'] = data['Area'].astype(str).apply(lambda x: re.findall('[0-9.]+', x))
    data['Area'] = data['Area'].apply(lambda x: ''.join(x))
    data['Area'] = pd.to_numeric(data['Area'], errors='coerce')

def str_columns(data):
    data['Street'] = data['Street'].astype(str).apply(lambda x: re.findall('[א-ת\s]+', x))
    data['Street'] = data['Street'].apply(lambda x: ''.join(x))
    data['city_area'] = data['city_area'].str.replace('[^\w\s]', '', regex=True)
    translator = str.maketrans('', '', string.punctuation)
    data['description'] = data['description'].str.translate(translator)
    data['City'] = data['City'].replace(' שוהם','שוהם')
    data['City'] = data['City'].replace('שוהם','שוהם')
    data['City'] = data['City'].replace(' נהרייה','נהריה')
    data['City'] = data['City'].replace('נהרייה','נהריה')
    data['condition'] = data['condition'].replace('דורש שיפוץ', 'ישן')
    data['condition'] = data['condition'].str.replace('FALSE', 'לא צויין')
    data['condition'] = data['condition'].str.replace('None', 'לא צויין')
    data['condition'] = data['condition'] .fillna('לא צויין')


def floors(data):
    data['floor'] = data['floor_out_of'].str.extract(r'(\d+)')
    data['floor'] = data['floor'].fillna('0').astype(int)
    data['total_floors'] = data['floor_out_of'].str.extract(r'(\d+)$')
    data['total_floors'] = data['total_floors'].fillna('0').astype(int)
    data['floor_out_of']

def entranceDate_Update(date):
    current_time = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(date, str):
        if date == 'גמיש' or date == 'גמיש ':
            return 'flexible'
        elif date == 'לא צויין':
            return 'not_defined'
        elif date == 'מיידי':
            return 'less_than_6 months'
        else:
            return date
    
    time_diff = (current_time - date).days
    
    if time_diff < 180:
        return 'less_than_6 months'
    elif 180 < time_diff < 365:
        return 'months_6_12'
    else:
        return 'above_year'
    
def boolean(data):
    columns_to_convert = ['hasElevator', 'hasParking', 'hasBars','hasStorage','hasAirCondition','hasBalcony','hasMamad','handicapFriendly']
    for column in columns_to_convert:
        
        data[column] = data[column].apply(lambda x: 0 if pd.isna(x) else (1 if x in [True, 'true'] else (0 if x in [False, 'false', 'אין', 'לא', 'no'] else (1 if any(word in str(x) for word in ['יש', 'כן', 'yes']) else 0))))
        
def typeUpdate(data):
    data.loc[:, 'type'] = data['type'].str.replace("'", "")
    data.loc[(data['type'] == 'דירת גן') & (data['floor_out_of'] == 'קומת קרקע'), 'type'] = 'בית פרטי'
    data.loc[data['type'] == 'דירת גן', 'type'] = 'דירה בבניין'
    data['type'] = data['type'].replace({
        'דירה': 'דירה בבניין',
        'בניין': 'דירה בבניין',
        'דירת גג': 'דירה בבניין',
        'דופלקס': 'דירה בבניין',
        'דירת נופש': 'בית פרטי',
        'קוטג': 'בית פרטי',
        'קוטג טורי': 'בית פרטי',
        'דו משפחתי': 'בית פרטי',
        'פנטהאוז': 'פנטהאוז',
        'מיני פנטהאוז': 'פנטהאוז',
        'מגרש': 'אחר',
        'נחלה': 'אחר',
        'טריפלקס': 'אחר'
    })
    value_to_drop = ['אחר']
    data = data[~data['type'].isin(value_to_drop)]
    return data

def drop_duplicates(data):
    duplicate_props = data.duplicated()
    all_duplicates = data[duplicate_props]
    data = data.drop_duplicates()
    return all_duplicates, data

def prepare_data(data):
    data.columns = data.columns.str.strip()
    price(data)
    area(data)
    room_number(data)
    str_columns(data)
    floors(data)
    data['entranceDate'] = data['entranceDate'].apply(entranceDate_Update)
    boolean(data)
    typeUpdate(data)
    drop_duplicates(data)
    cols = ['City','price','type','room_number','Area','hasElevator','hasParking','hasMamad']
    model_data = data[cols]
    return model_data
prepare_data(data)

