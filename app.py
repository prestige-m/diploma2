import os
from flask import Flask,jsonify,request,render_template



app = Flask(__name__)

app.config.from_object('config')
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
  return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,
            use_reloader=True,
            port=4000)
