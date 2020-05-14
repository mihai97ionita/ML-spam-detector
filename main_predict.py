from functions import youtube_downloader
import pickle
# import things
from flask_table import Table, Col

def predict_in_html_table(videoId):
    # Load model
    with open(".\\models\\" + "ACTIVE") as f:
        active_model_file_name = f.readline()
    print("This is the model that we are going to use: "+active_model_file_name)
    model_file = open(".\\models\\" + active_model_file_name, "rb")
    model = pickle.load(model_file)
    model_file.close()

    # Load bags of words
    saved_file = open("Vectorizer.pkl","rb")
    Vectorizer_loaded=pickle.load(saved_file)
    saved_file.close()

    # download comments videoId
    #'K0KV7F4shEk'
    comments_table = youtube_downloader.download(videoId, 'comments_dump.json', 100)
    for index, comment in enumerate(comments_table['Comment']):
            Comment = [comment]
            bag_of_words = Vectorizer_loaded.transform(Comment).toarray()
            y_pred = model.predict(bag_of_words)
            comments_table.at[index, 'Predict'] = y_pred[0]
    print(comments_table)


    # Declare your table
    class ItemTable(Table):
        predict = Col('Predict')
        author = Col('Author')
        comment = Col('Comment')


    # Get some objects
    class Item(object):
        def __init__(self, predict, author, comment):
            self.predict = predict
            self.author = author
            self.comment = comment


    comments_table_html = []

    for index, rows in comments_table.iterrows():
        item = Item(rows[2].__str__(),
                    rows[0].__str__(),
                    rows[1].__str__())
        comments_table_html.append(item)


    print(comments_table_html)
    # Populate the table
    table = ItemTable(comments_table_html)

    # Print the html
    #print(table.__html__())
    # or just {{ table }} from within a Jinja template
    return table.__html__().replace('<tr><td>1','<tr style="background-color:#FF0000"><td>1')
