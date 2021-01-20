from flask_table import Table, Col
from flask import Flask
from classificator_package import classificator_service

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


def render_ui(list_of_comments):
    list_of_rows = []

    for element in list_of_comments:
        item = Item(predict=element['predict'],
                    author=element['text'],
                    comment=element['author'])
        list_of_rows.append(item)

    table = ItemTable(list_of_rows)

    return table.__html__().replace('<tr><td>1', '<tr style="background-color:#FF0000"><td>1')

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def render_home():
    return "\n Put your VideoID in the path of the website :D ^^\n Like this /K0KV7F4shEk"


@app.route('/<video_id>')
def run_spam_detector(video_id: str):
    predict_comments = classificator_service.display_comments(video_id)
    return render_ui(predict_comments)


if __name__ == '__main__':
    app.run()
