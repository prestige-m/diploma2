from flask import Blueprint


recognition = Blueprint('recognition', __name__, template_folder='templates')
from app.face_recognize import routes