import flask
import pickle
import requests
from microservice_train_store import train
#from microservice_data_processing import load_dataset

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/train', methods=['GET'])
def activate_train():
    response = requests.get('http://localhost:8080/data')
    load_dataset = pickle.loads(response)
    results= train.DT_boosted_fit(load_dataset.x_train, load_dataset.y_train, load_dataset.x_test, load_dataset.y_test, "ALL")
    return "New model trained:\n{}".format(results)


@app.route('/model', methods=['GET'])
def get_active_model():
    with open("../models/" + "ACTIVE") as f:
        active_model_file_name = f.readline()
    model_file = open("models/{}".format(active_model_file_name), "rb")
    return model_file


app.run(port=8081)
