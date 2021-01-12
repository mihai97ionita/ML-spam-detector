from microservice_comments_extractor import youtube_downloader
import pickle
from expiringdict import ExpiringDict

# added small cache for 1 min
cache = ExpiringDict(max_len=5, max_age_seconds=60)


def predict(videoId):
    if cache.get(videoId) is None:
        # Load model
        list_of_dicts = predict_computation(videoId)
        cache[videoId] = list_of_dicts
        return list_of_dicts
    else:
        print(f"Using cache for videoId {videoId} ")
        return cache.get(videoId)


def predict_computation(videoId):
    with open("../models/" + "ACTIVE") as f:
        active_model_file_name = f.readline().replace('\n', '')
    print("This is the model that we are going to use: " + active_model_file_name)
    model_file = open(f"../models/{active_model_file_name}", "rb")
    model = pickle.load(model_file)
    model_file.close()
    # Load bags of words
    saved_file = open("../Vectorizer.pkl", "rb")
    Vectorizer_loaded = pickle.load(saved_file)
    saved_file.close()
    # download comments videoId
    list_of_dicts = youtube_downloader.comments_extractor(videoId)
    for index, comment in enumerate(list_of_dicts):
        Comment = [comment['text']]
        bag_of_words = Vectorizer_loaded.transform(Comment).toarray()
        y_pred = model.predict(bag_of_words)
        comment['predict'] = y_pred[0]
    return list_of_dicts
