
from fastapi import FastAPI, Response, Request, Body
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import tensorflow as tf

import rasterio as rio

import joblib
import pickle
import pyproj
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
categorical_cols = ['Walls', 'Roof', 'Pillars']

model_name = "map_model\Flood_map_model.keras"
model = tf.keras.models.load_model(model_name)
model = None

damage_model = joblib.load('model/xg_model.joblib')
damage_encoder = joblib.load('model/encoder.joblib')
with open('model/transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/profile_data.pkl', 'rb') as f:
    profile = pickle.load(f)
source_crs = 'epsg:4326'  # Global lat-lon coordinate system
target_crs = 'epsg:5235'  # Coordinate system of the file
latlon_to_polar = pyproj.Transformer.from_crs(source_crs, target_crs)


def get_pred_flie():

    filename = 'map_gen.asc' if os.path.exists(
        'map_gen.asc') else 'map_test.wd'

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
    image = get_pred_flie()

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
    flood_map = get_pred_flie()
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
    flood_map = get_pred_flie()
    lat, lon = input_data['lat'], input_data['lon']
    cordx, cordy = latlon_to_polar.transform(lat, lon)
    x, y = transformer.rowcol(cordx, cordy)

    # send bad request if the location is out of bounds
    if x < 0 or y < 0 or x >= flood_map.shape[0] or y >= flood_map.shape[1]:
        return Response(content="Location out of bounds", media_type="text/plain", status_code=400)
    flood_height = flood_map[x, y]

    return {'flood_height': float(flood_height)}


@app.post("/generate/")
async def generate(request: Request, input_data: dict = Body(...)):
    upstream1 = input_data['upstream1']
    upstream2 = input_data['upstream2']
    downstream1 = input_data['downstream1']
    downstream2 = input_data['downstream2']

    upstream_interpolate = np.linspace(upstream1, upstream2, num=9)
    downstream_interpolate = np.linspace(downstream1, downstream2, num=9)
    stream_data = [upstream_interpolate[0], downstream_interpolate[0]] + \
        list(upstream_interpolate[1:])+list(downstream_interpolate[1:])
    stream_df = pd.DataFrame([stream_data])

    stream_scaled = scaler.transform(stream_df)
    stream_scaled = stream_scaled.reshape(1, 1, 18)
    pred = model.predict(stream_scaled)
    pred.resize(600, 900)
    pred[pred < 0.15] = 0
    with rio.Env():
        profile.update(dtype=str(pred.dtype), count=1, compress='lzw')
        with rio.open("map_gen.asc", 'w', **profile) as dst:
            dst.write(pred, 1)

    return {"image": pred.tolist(), "max_height": int(pred.max())}
