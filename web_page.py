from flask import Flask, render_template
import main_predict
app = Flask(__name__)


@app.route('/')
def hello():
    return "\n Put your VideoID in the path of the website :D ^^\n Like /K0KV7F4shEk "

@app.route('/<videoId>')
def table(videoId):
    #table = open(".\\html_tables\\" + "table" + ".html", 'w+')
    page = main_predict.predict_in_html_table(videoId)
    return page

if __name__ == '__main__':
    app.run()
