from flask import Flask
from microservice_classify import main_predict
from flask_table import Table, Col

app = Flask(__name__)
app.config["DEBUG"] = True


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


def to_html_visual(list_of_comments):
    list_of_rows = []

    for element in list_of_comments:
        item = Item(predict=element['predict'],
                    author=element['text'],
                    comment=element['author'])
        list_of_rows.append(item)

    table = ItemTable(list_of_rows)

    return table.__html__().replace('<tr><td>1', '<tr style="background-color:#FF0000"><td>1')


@app.route('/')
def render_home():
    return "\n Put your VideoID in the path of the website :D ^^\n Like /K0KV7F4shEk"


@app.route('/<videoId>')
def render_predict_results(video_id: str):
    predict_results = main_predict.predict(video_id)
    return to_html_visual(predict_results)


if __name__ == '__main__':
    app.run()
