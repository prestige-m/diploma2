##   app/__init__.py  ##
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_login import LoginManager
from flask_bcrypt import Bcrypt

db = SQLAlchemy()
bootstrap = Bootstrap()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'authentication.login'
login_manager.session_protection = 'strong'

def create_app():
    app = Flask(__name__)
    configuration = os.path.join(os.getcwd(), "config.py")
    app.config.from_pyfile(configuration)
    db.init_app(app)
    bootstrap.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)

    from app.auth import authentication
    from app.face_recognize import recognition

    app.register_blueprint(authentication)
    app.register_blueprint(recognition)

    app.config['ROOT_PATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['UPLOAD_FOLDER'] = os.path.join(app.config['ROOT_PATH'], "uploads")
    app.config['DATASET_FOLDER'] = os.path.join(app.config['ROOT_PATH'], "dataset")
    app.config['MODELS_FOLDER'] = os.path.join(app.config['ROOT_PATH'], "models")

    return app