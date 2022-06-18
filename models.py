import os
from unicodedata import category
import sqlalchemy as sa
from app import db
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import UserMixin
from flask import url_for

book_category = db.Table('book_category',
    db.Column('book_id', db.Integer, db.ForeignKey('books.id')),
    db.Column('category_id', db.Integer, db.ForeignKey('categories.id'))
)

class Category(db.Model):
    __tablename__ = 'categories'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('categories.id'))

    def __repr__(self):
        return '<Category %r>' % self.id

# class Logs(db.Model):
#     __tablename__ = 'logs'

#     id = db.Column(db.Integer, primary_key=True)
#     created_at = db.Column(db.DateTime, nullable=False, server_default=sa.sql.func.now())
#     user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
#     book_id= db.Column(db.Integer, db.ForeignKey('books.id'))
#     def __repr__(self):
#         return '<Logs %r>' % self.id

class Roles(db.Model):
    __tablename__ = 'roles'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    desc = db.Column(db.Text)

    def __repr__(self):
        return '<roles %r>' % self.name

class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    last_name = db.Column(db.String(100), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    middle_name = db.Column(db.String(100))
    login = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=sa.sql.func.now())
    role =  db.Column(db.Integer, db.ForeignKey('roles.id'))

    def __repr__(self):
        return '<User %r>' % self.login

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def create_user():
        user = User(first_name='Иван', last_name='Иванов', login='user',role="1")
        user.set_password('qwerty')
        db.session.add(user)
        db.session.commit()

    @property
    def full_name(self):
        return ' '.join([self.last_name, self.first_name, self.middle_name or ''])


class Book(db.Model):
    __tablename__ = 'books'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    short_desc = db.Column(db.Text(), nullable=False)
    rating_sum = db.Column(db.Integer, nullable=False, default=0)
    rating_num = db.Column(db.Integer, nullable=False, default=0)
    year = db.Column(db.Integer,nullable=False)
    author = db.Column(db.String(100), nullable=False)
    pub_house = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=sa.sql.func.now())
    volume = db.Column(db.Integer, nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'))#, ondelete='CASCADE')
    view = db.Column(db.Integer, nullable=False, default=0)
    bg_image = db.relationship('Image')
    categories = db.relationship('Category', secondary=book_category,
        backref=db.backref('books'))

    @property
    def rating(self):
        if self.rating_num > 0:
            return self.rating_sum / self.rating_num
        return 0

    def addmark(self,mark):
       self.rating_sum =self.rating_sum+mark
       self.rating_num =self.rating_num+1

    def __repr__(self):
        return '<Book %r>' % self.name


class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(100), nullable=False)
    mime_type = db.Column(db.String(100), nullable=False)
    md5_hash = db.Column(db.String(256), nullable=False, unique=True)
    object_type = db.Column(db.String(100))
    book_id = db.Column(db.Integer)
    active = db.Column(db.Boolean, nullable=False, default=False)

    def __repr__(self):
        return '<Image %r>' % self.file_name

    @property
    def storage_filename(self):
        _, ext = os.path.splitext(self.file_name)
        return str(self.id) + ext

    @property
    def url(self):
        return url_for('image', image_id=self.id)

class Review(db.Model):
    __tablename__ = 'reviews'

    id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text(), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=sa.sql.func.now())
    book_id = db.Column(db.Integer, db.ForeignKey('books.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    book = db.relationship('Book')
    user = db.relationship('User')

    def __repr__(self):
        return '<Review %r>' % self.id
