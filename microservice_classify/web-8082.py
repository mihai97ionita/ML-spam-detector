# import flask
# import pickle
# import requests
# from flask import request
#
# from microservice_train_store import train
# #from microservice_data_processing import load_dataset
#
# app = flask.Flask(__name__)
# app.config["DEBUG"] = True
#
#
# @app.route('/clasify', methods=['GET'])
# def get_clasify():
#     response = requests.get('http://localhost:8081/model')
#     model = pickle.loads(response)
#
#     query_parameters = request.args
#     id = query_parameters.get('id')
#     return model_file
#
#
# app.run(port=8082)
