from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def get_current_time():
    data = request.data
    request.headers['Content-Type'] = 'application/json'
    return "yelpyelpyelp" + data.decode("utf-8")

