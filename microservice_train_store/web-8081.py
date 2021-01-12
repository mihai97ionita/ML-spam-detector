import json

import flask
import pickle
import requests
from flask import send_from_directory

from microservice_train_store import train
#from microservice_data_processing import load_dataset

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/train', methods=['GET'])
def activate_train():
    my_response_json = requests.get('http://localhost:8080/data').json()
    my_response = json.loads(my_response_json)
    load_dataset = pickle.loads(my_response)
    results = train.DT_boosted_fit(load_dataset.x_train, load_dataset.y_train, load_dataset.x_test, load_dataset.y_test, "ALL")
    return f"New model trained:\n{results}"


@app.route('/model', methods=['GET'])
def get_active_model():
    with open("../models/" + "ACTIVE") as f:
        active_model_file_name = f.readline()
    model_file = open(f"../models/{active_model_file_name}", "rb")
    return send_from_directory()


app.run(port=8081)
