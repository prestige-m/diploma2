from flask import render_template, request, flash, redirect, url_for
from app.auth.forms import RegistrationForm, LoginForm
from app.auth import authentication
from app.auth.models import User, Counter
from flask_login import login_user, logout_user, login_required, current_user


@authentication.route('/')
def route_default():
    return redirect(url_for('authentication.login'))


@authentication.route('/register', methods=['GET', 'POST'])
def register():

    if current_user.is_authenticated:
        flash('Ви вже авторизовані!', "warning")
        return redirect(url_for('authentication.homepage'))
    form = RegistrationForm()

    if form.validate_on_submit():

        user = User.query.filter_by(email=form.email.data).first()
        if user:
            flash("Користувач з такою електронною адресою вже існує!", "error")
            return render_template('registration.html', form=form, success=False)

        User.create_user(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data
        )
        flash("Реєстрація успішна!", "success")
        return redirect(url_for('authentication.homepage'))
        
    return render_template('registration.html', form=form)

@authentication.route('/')
def index():
    counter_value = Counter.get()
    return render_template('index.html', counter_value=counter_value)

@authentication.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        flash('Ви вже авторизовані!', "warning")
        return redirect(url_for('authentication.homepage'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if not user or not user.check_password(form.password.data):
            flash('Введено некоректні дані!', "error")
            return redirect(url_for('authentication.login'))
        
        login_user(user, form.stay_loggedin.data)
        return(redirect(url_for('authentication.homepage')))
        
    return render_template('login.html', form=form)


@authentication.route('/homepage')
def homepage():
    counter_value = Counter.get()
    return render_template('index.html', counter_value=counter_value)


@authentication.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('authentication.login'))

@authentication.app_errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

