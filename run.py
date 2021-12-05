from app import create_app, db
from app.auth.models import User

#if __name__ == '__main__':
flask_app = create_app('prod')
with flask_app.app_context():
    db.create_all()
    if not User.query.filter_by(username='test').first():
        User.create_user(username='test',
                            email='test@test.com',
                            password='test')
flask_app.run(port=4111)
#flask_app.run(ssl_context=('cert.pem', 'key.pem'))