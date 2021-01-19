from flask import Flask
from classificator_package import classificator_service
from spam_detector_package import spam_detector_service

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def render_home():
    return "\n Put your VideoID in the path of the website :D ^^\n Like /K0KV7F4shEk"


@app.route('/<video_id>')
def run_spam_detector(video_id: str):
    predict_comments = classificator_service.display_comments(video_id)
    return spam_detector_service.run_spam_detector(predict_comments)


if __name__ == '__main__':
    app.run()
