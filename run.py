import os
from app import create_app, db
from app.auth.models import User

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flask_app = create_app()
with flask_app.app_context():
    db.create_all()
    if not User.query.filter_by(username='test').first():
        User.create_user(username='test',
                            email='test@test.com',
                            password='test')

flask_app.run(port=4111)
