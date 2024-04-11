from typing import Union

from fastapi import FastAPI, Response, Request, Body
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
from scipy import ndimage
from io import BytesIO
from PIL import Image
import rasterio as rio
import matplotlib.pyplot as plt
import joblib


def load_model(name):
    # Loading the model weights
    from tensorflow.keras.models import model_from_json

    # Model reconstruction from JSON file
    with open(name+'.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
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
damage_model = joblib.load('svm_model.joblib')
damage_encoder = joblib.load('encoder.joblib')


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
    y_pred.resize(611, 951)
    y_pred[y_pred < 0.2] = 0

    return y_pred


def get_pred2():
    filename = 'map_test.wd'

    with rio.open(filename) as real_data:
        img_data = real_data.read(1)

        return img_data


def get_pred_rand():
    x_test = np.random.rand(28)
    x_test = x_test.reshape(1, 1, 28)

    # y_pred = model.predict(x_test)
    # y_pred.resize(611, 951)
    # y_pred[y_pred < 0.2] = 0
    y_pred = np.random.rand(900, 500)

    return y_pred


def bin_estimated_value(estimated_value):
    value_bins = [0, 2000000, 4000000, 6000000, 8000000, 10000000,
                  12000000, 14000000, 16000000, 18000000, 20000000, float('inf')]
    return pd.cut([estimated_value], bins=value_bins).codes[0]


def get_estimated_value_range(value):
    repair_bins = [0, 20000, 40000, 60000,
                   80000, 100000]
    if value == 0:
        return f"0 - {repair_bins[value+1]}"
    elif value == len(repair_bins)-1:
        return f"{repair_bins[value]} or more"
    else:
        return f"{repair_bins[value]} - {repair_bins[value+1]}"


@app.get("/")
def read_root():
    # y_pred = get_pred()
    image = get_pred2()
    image[image < 0] = 0
    print(image.shape)

    new_shape = (500, 500)

    # Resize the image using bilinear interpolation (recommended for scaling down)
    resized_image = ndimage.zoom(image, zoom=(new_shape[0] / image.shape[0],
                                              new_shape[1] / image.shape[1]),
                                 mode='reflect')
    print(resized_image.shape)
    # return {"image": [[1, 2, 3], [1, 2, 3]]}
    return {"image": resized_image.tolist(), "max_height": int(resized_image.max())}


@app.get("/image", response_class=Response)
def get_image():
    img_data = get_pred_rand()

    fig, ax = plt.subplots()
    ax.imshow(img_data)
    ax.axis('off')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='jpeg')
    plt.close(fig)

    image_bytes = buffer.getvalue()

    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/image2", response_class=Response)
def get_image2():
    filename = 'data/Run2-0073.wd'

    with rio.open(filename) as real_data:
        img_data = real_data.read(1)

        fig, ax = plt.subplots()
        ax.imshow(img_data)
        ax.axis('off')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='jpeg')
        plt.close(fig)

        image_bytes = buffer.getvalue()

        return Response(content=image_bytes, media_type="image/jpeg")


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
        'Flood Height': input_data['Flood Height']
    }, index=[0])

    # Encode categorical columns in the input data
    encoded_data = damage_encoder.transform(input_df[categorical_cols])
    encoded_cols = damage_encoder.get_feature_names_out(categorical_cols)
    encoded_data_df = pd.DataFrame(
        encoded_data.toarray(), columns=encoded_cols)

    data_numerical = input_df.drop(columns=categorical_cols)
    data_final = pd.concat([encoded_data_df, data_numerical], axis=1)

    predictions = damage_model.predict(data_final)

    final_predictions = get_estimated_value_range(predictions[0])
    return {'predicted_class': final_predictions}


@app.post("/predict_loc/")
async def predict(request: Request, input_data: dict = Body(...)):
    flood_map = get_pred_rand()
    x, y = input_data['x'], input_data['y']
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
        'Flood Height': flood_height
    }, index=[0])

    # Encode categorical columns in the input data
    encoded_data = damage_encoder.transform(input_df[categorical_cols])
    encoded_cols = damage_encoder.get_feature_names_out(categorical_cols)
    encoded_data_df = pd.DataFrame(
        encoded_data.toarray(), columns=encoded_cols)

    data_numerical = input_df.drop(columns=categorical_cols)
    data_final = pd.concat([encoded_data_df, data_numerical], axis=1)

    predictions = damage_model.predict(data_final)

    final_predictions = get_estimated_value_range(predictions[0])
    return {'predicted_class': final_predictions}
