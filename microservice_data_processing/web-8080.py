import flask
import pickle
from microservice_data_processing import load_dataset

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/data', methods=['GET'])
def get_data():
    return pickle.dumps(load_dataset)


app.run(port=8080)
