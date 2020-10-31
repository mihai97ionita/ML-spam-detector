from youtube_extractor_microservice import youtube_downloader
import pickle
# import things


def predict(videoId):
    # Load model
    with open("../models/" + "ACTIVE") as f:
        active_model_file_name = f.readline()
    print("This is the model that we are going to use: "+active_model_file_name)
    model_file = open("../models/Decision trees Classifier 2020-05-03T17 10 02.pkl", "rb")
    model = pickle.load(model_file)
    model_file.close()

    # Load bags of words
    saved_file = open("../Vectorizer.pkl", "rb")
    Vectorizer_loaded = pickle.load(saved_file)
    saved_file.close()

    # download comments videoId
    comments_table = youtube_downloader.download(videoId, 'comments_dump.json', 100)
    for index, comment in enumerate(comments_table['Comment']):
            Comment = [comment]
            bag_of_words = Vectorizer_loaded.transform(Comment).toarray()
            y_pred = model.predict(bag_of_words)
            comments_table.at[index, 'Predict'] = y_pred[0]
    return comments_table
