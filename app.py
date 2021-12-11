import os
from flask import Flask


app = Flask(__name__)

app.config.from_object('config')
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = os.path.basename('dataset')


if __name__ == '__main__':
    app.run(debug=True,
            use_reloader=True,
            port=4000)
