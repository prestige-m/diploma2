from datetime import datetime
from app import db, bcrypt
from app import login_manager
from flask_login import UserMixin

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    #user_public_id = db.Column(db.String(60))
    username = db.Column(db.String(64))
    email = db.Column(db.String(64), unique=True, index=True)
    password = db.Column(db.String(80))
    create_date = db.Column(db.DateTime, default=datetime.now)

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

    @classmethod
    def create_user(cls, username, email, password):

        user = cls(username=username,
                   email=email,
                   password=bcrypt.generate_password_hash(password).decode('utf-8')
            )
        db.session.add(user)
        db.session.commit()
        return user

    def __repr__(self):
        return str(self.username)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    return user if user else None


class Counter(db.Model):
    __tablename__ = 'counter'

    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Integer)

    @classmethod
    def update(cls):
        counter = Counter.query.get(1)
        counter.value = 1 + counter.value
        db.session.commit()

    @classmethod
    def get(cls):
        counter = Counter.query.get(1)
        return counter.value

