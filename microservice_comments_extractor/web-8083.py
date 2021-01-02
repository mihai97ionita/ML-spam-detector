import flask
import pickle
import requests
from flask import request
from microservice_comments_extractor import youtube_downloader

from microservice_train_store import train
#from microservice_data_processing import load_dataset

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/comments', methods=['GET'])
def get_comments_from_url():
    query_parameters = request.args
    videoId = query_parameters.get('id')
    return youtube_downloader.comments_extractor(videoId)


app.run(port=8083)
