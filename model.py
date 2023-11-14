import os
from flask import Flask, request, send_from_directory
import keras

app = Flask(__name__, static_folder='./build', static_url_path='/')



@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def main_page():
    if request.method == 'POST':
        json_data = request.get_json()['data']
        print([json_data])
        model = keras.models.load_model('./model.keras')
    return "hello"
