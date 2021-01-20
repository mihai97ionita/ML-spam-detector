from classificator_package.comment_extractor import youtube_downloader_api
import pickle
from expiringdict import ExpiringDict

# added small cache for 1 min
from train_store_package import train_store_service

cache = ExpiringDict(max_len=5, max_age_seconds=60)


def display_comments(video_id: str):
    if cache.get(video_id) is None:
        list_of_dicts = get_comments(video_id)
        cache[video_id] = list_of_dicts
        return list_of_dicts
    else:
        print(f"Using cache for videoId {video_id} ")
        return cache.get(video_id)


def try_load_trained_model_and_bag_of_words():
    model_file = train_store_service.load_trained_model()
    model = pickle.load(model_file)
    model_file.close()
    # this data is saved by train and store
    saved_file = train_store_service.load_bag_of_words()
    bag_of_words = pickle.load(saved_file)
    saved_file.close()
    # download comments videoId
    return model, bag_of_words


def load_trained_model_and_bag_of_words():
    try:
        # if we can't load model and bag of words, we need to train it
        model, bag_of_words = try_load_trained_model_and_bag_of_words()
    except Exception:
        train_store_service.train()
        model, bag_of_words = try_load_trained_model_and_bag_of_words()
    return model, bag_of_words


def get_comments(video_id: str):
    model, bag_of_words = load_trained_model_and_bag_of_words()
    list_of_dicts_of_comments = youtube_downloader_api.parse(video_id)
    for index, comment in enumerate(list_of_dicts_of_comments):
        parsed_comment = bag_of_words.transform([comment['text']]).toarray()
        y_predict = model.predict(parsed_comment)
        comment['predict'] = y_predict[0]
    return list_of_dicts_of_comments

