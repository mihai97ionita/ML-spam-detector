from flask_table import Table, Col


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


def compute_ui(list_of_comments):
    list_of_rows = []

    for element in list_of_comments:
        item = Item(predict=element['predict'],
                    author=element['text'],
                    comment=element['author'])
        list_of_rows.append(item)

    table = ItemTable(list_of_rows)

    return table.__html__().replace('<tr><td>1', '<tr style="background-color:#FF0000"><td>1')
