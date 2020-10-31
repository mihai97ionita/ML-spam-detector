from flask import Flask
from predict_microservice import main_predict
from flask_table import Table, Col
app = Flask(__name__)


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


def to_html_visual(comments_table):
    comments_table_html = []

    for index, rows in comments_table.iterrows():
        item = Item(rows[2].__str__(),
                    rows[0].__str__(),
                    rows[1].__str__())
        comments_table_html.append(item)

    table = ItemTable(comments_table_html)

    return table.__html__().replace('<tr><td>1','<tr style="background-color:#FF0000"><td>1')


@app.route('/')
def hello():
    return "\n Put your VideoID in the path of the website :D ^^\n Like /K0KV7F4shEk"


@app.route('/<videoId>')
def table(video_id):
    predict_results = main_predict.predict(video_id)
    page = to_html_visual(predict_results)
    return page


if __name__ == '__main__':
    app.run()
