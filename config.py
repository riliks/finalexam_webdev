from pickle import TRUE
import os

SECRET_KEY = '8eafd8fcb5bb15060ed481856d95b84f36ccde62755f9fb489e7f46ac14129d6'
MYSQL_USER = 'std_1670_exam'
MYSQL_HOST = 'std-mysql.ist.mospolytech.ru'
MYSQL_DATABASE = 'std_1670_exam'
MYSQL_PASSWORD = '12345678'
SQLALCHEMY_DATABASE_URI = f'mysql+mysqlconnector://std_1670_exam:12345678@std-mysql.ist.mospolytech.ru/std_1670_exam'
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = False

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', 'images')