from classificator_package.comment_extractor import youtube_downloader
import pickle
from expiringdict import ExpiringDict

# added small cache for 1 min
from train_store_package import train_store_service

cache = ExpiringDict(max_len=5, max_age_seconds=60)


def predict(video_id: str):
    if cache.get(video_id) is None:
        # Load model
        list_of_dicts = predict_computation(video_id)
        cache[video_id] = list_of_dicts
        return list_of_dicts
    else:
        print(f"Using cache for videoId {video_id} ")
        return cache.get(video_id)


def try_get_model_and_bag_of_words():
    with open("../train_store_package/models/" + "ACTIVE") as f:
        active_model_file_name = f.readline().replace('\n', '')
        if active_model_file_name == "" or active_model_file_name is None:
            raise Exception("No file found, use train service")
    print("This is the model that we are going to use: " + active_model_file_name)
    model_file = open(f"../train_store_package/models/{active_model_file_name}", "rb")
    model = pickle.load(model_file)
    model_file.close()
    # Load bags of words
    saved_file = open("../train_store_package/bag_of_words.pkl", "rb")
    bag_of_words = pickle.load(saved_file)
    saved_file.close()
    # download comments videoId
    return model, bag_of_words


def get_model_and_bag_of_words():
    try:
        # if we can't load model and bag of words, we need to train it and generate bag of words
        model, bag_of_words = try_get_model_and_bag_of_words()
    except Exception:
        train_store_service.train_store()
        model, bag_of_words = try_get_model_and_bag_of_words()
    return model, bag_of_words


def predict_computation(video_id: str):
    model, bag_of_words = get_model_and_bag_of_words()
    # download comments videoId
    list_of_dicts_of_comments = youtube_downloader.comments_extractor(video_id)
    for index, comment in enumerate(list_of_dicts_of_comments):
        parsed_comment = bag_of_words.transform([comment['text']]).toarray()
        y_predict = model.predict(parsed_comment)
        comment['predict'] = y_predict[0]
    return list_of_dicts_of_comments
