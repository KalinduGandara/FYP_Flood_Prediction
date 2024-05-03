
from fastapi import FastAPI, Response, Request, Body
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd

import rasterio as rio

import joblib
import pickle
import pyproj


def load_model(name):
    from tensorflow.keras.models import model_from_json

    with open(name+'.json', 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(name+'.h5')

    return model


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
categorical_cols = ['Walls', 'Roof', 'Pillars']

model_name = "map_model/CNN_Model_2005"
# model = load_model(model_name)
model = None

damage_model = joblib.load('model/xg_model.joblib')
damage_encoder = joblib.load('model/encoder.joblib')
with open('model/transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)
source_crs = 'epsg:4326'  # Global lat-lon coordinate system
target_crs = 'epsg:5235'  # Coordinate system of the file
latlon_to_polar = pyproj.Transformer.from_crs(source_crs, target_crs)


def get_pred():
    data = [-1.64682254, -1.11010161, -1.06129654, -1.1050725, -1.10742274,
            -1.10469143, -1.10184463, -1.09881952, -1.09553624, -1.09196258,
            -1.34811639, -1.4042805, -1.06090403, -1.05953386, -1.05719865,
            -1.05473959, -1.05302812, -1.05292771, -1.26244903, -1.2782434,
            -1.10653672, -1.11049255, -1.11185931, -1.11254004, -1.1157182,
            -1.11330916, -1.26749804, -1.26498577]
    x_test = np.array(data)
    x_test = x_test.reshape(1, 1, 28)

    y_pred = model.predict(x_test)
    y_pred.resize(600, 900)
    y_pred[y_pred < 0.15] = 0

    return y_pred


def get_pred2():
    filename = 'map_test.wd'

    with rio.open(filename) as real_data:
        img_data = real_data.read(1)

        return img_data


def bin_estimated_value(estimated_value):
    value_bins = [0, 2000000, 4000000, 6000000, 8000000, 10000000,
                  12000000, 14000000, 16000000, 18000000, 20000000, float('inf')]
    return pd.cut([estimated_value], bins=value_bins).codes[0]


def get_estimated_value_range(value):
    repair_bins = [0, 20000, 40000, 60000,
                   80000, 100000]
    if value == 0:
        return "No damage"
    elif value == len(repair_bins):
        return f"{repair_bins[value-1]} or more"
    else:
        return f"{repair_bins[value-1]} - {repair_bins[value]}"


def bin_flood_height(height):
    height_bins = [-float('inf'), 1, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225,
                   240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, float('inf')]
    return pd.cut([height], bins=height_bins).codes[0]


@app.get("/")
def read_root():
    # image = get_pred()
    image = get_pred2()

    return {"image": image.tolist(), "max_height": int(image.max())}


@app.post("/predict/")
async def predict(request: Request, input_data: dict = Body(...)):
    # Create a DataFrame with user input
    input_df = pd.DataFrame({
        'Building Age': input_data['Building Age'],
        'Estimated value bins': bin_estimated_value(input_data['Estimated value']),
        'Floors': input_data['Floors'],
        'Building Height': input_data['Building Height'],
        'Walls': input_data['Walls'],
        'Roof': input_data['Roof'],
        'Pillars': input_data['Pillars'],
        'Flood Height bins': bin_flood_height(input_data['Flood Height'])
    }, index=[0])
    # Encode categorical columns in the input data
    encoded_data = damage_encoder.transform(input_df[categorical_cols])
    encoded_cols = damage_encoder.get_feature_names_out(categorical_cols)
    encoded_data_df = pd.DataFrame(
        encoded_data.toarray(), columns=encoded_cols)

    data_numerical = input_df.drop(columns=categorical_cols)
    data_final = pd.concat([encoded_data_df, data_numerical], axis=1)

    predictions = damage_model.predict(data_final.values)

    final_predictions = get_estimated_value_range(predictions[0])
    return {'predicted_class': final_predictions}


@app.post("/predict_loc/")
async def predict(request: Request, input_data: dict = Body(...)):
    flood_map = get_pred2()
    lat, lon = input_data['lat'], input_data['lon']
    cordx, cordy = latlon_to_polar.transform(lat, lon)
    x, y = transformer.rowcol(cordx, cordy)

    # send bad request if the location is out of bounds
    if x < 0 or y < 0 or x >= flood_map.shape[0] or y >= flood_map.shape[1]:
        return Response(content="Location out of bounds", media_type="text/plain", status_code=400)
    flood_height = flood_map[x, y]

    # Create a DataFrame with user input
    input_df = pd.DataFrame({
        'Building Age': input_data['Building Age'],
        'Estimated value bins': bin_estimated_value(input_data['Estimated value']),
        'Floors': input_data['Floors'],
        'Building Height': input_data['Building Height'],
        'Walls': input_data['Walls'],
        'Roof': input_data['Roof'],
        'Pillars': input_data['Pillars'],
        'Flood Height bins': bin_flood_height(flood_height)
    }, index=[0])
    # Encode categorical columns in the input data
    encoded_data = damage_encoder.transform(input_df[categorical_cols])
    encoded_cols = damage_encoder.get_feature_names_out(categorical_cols)
    encoded_data_df = pd.DataFrame(
        encoded_data.toarray(), columns=encoded_cols)

    data_numerical = input_df.drop(columns=categorical_cols)
    data_final = pd.concat([encoded_data_df, data_numerical], axis=1)

    predictions = damage_model.predict(data_final.values)

    final_predictions = get_estimated_value_range(predictions[0])
    return {'predicted_class': final_predictions}


@app.post("/height/")
async def height(request: Request, input_data: dict = Body(...)):
    flood_map = get_pred2()
    lat, lon = input_data['lat'], input_data['lon']
    cordx, cordy = latlon_to_polar.transform(lat, lon)
    x, y = transformer.rowcol(cordx, cordy)

    # send bad request if the location is out of bounds
    if x < 0 or y < 0 or x >= flood_map.shape[0] or y >= flood_map.shape[1]:
        return Response(content="Location out of bounds", media_type="text/plain", status_code=400)
    flood_height = flood_map[x, y]

    return {'flood_height': float(flood_height)}
