import hashlib
from unicodedata import category
import uuid
import os
from werkzeug.utils import secure_filename
from models import Book, Image,Category
from app import db, app
from flask import flash
import datetime
now = datetime.datetime.now()



def check_text(text,flag=''):
    if flag=='year':
        if text=='':
            text=0
        if 100<int(text)<=now.year:
            return False
        else:
            return 'Некорректная дата'
    if not text:
        return 'Поле не может быть пустым'
    else:
        return False
                
class BooksFilter:
    def __init__(self, name, category_ids):
        self.name = name
        self.category_ids = category_ids
        self.query = Book.query
        flash(category_ids)
    def perform(self):
        self.__filter_by_name()
        self.__filter_by_category_ids()
        return self.query.order_by(Book.created_at.desc())

    def __filter_by_name(self):
        if self.name:
            self.query = self.query.filter(Book.name.ilike('%' + self.name + '%'))

    def __filter_by_category_ids(self):
        if self.category_ids:
            for book in Book.query.all():
                for cat in book.categories:
                    flash(book)
                    if cat.id==self.category_ids:
                        pass
                        # self.query.append=book
       

class ImageSaver:
    def __init__(self, file):
        self.file = file

    def save(self):
        self.img = self.__find_by_md5_hash()
        if self.img is not None:
            return self.img
        file_name = secure_filename(self.file.filename)
        self.img = Image(
            file_name=file_name,
            mime_type=self.file.mimetype,
            md5_hash=self.md5_hash)

        db.session.add(self.img)
        db.session.commit()

        self.file.save(
            os.path.join(app.config['UPLOAD_FOLDER'],
                         self.img.storage_filename))
        return self.img

    def __find_by_md5_hash(self):
        self.md5_hash = hashlib.md5(self.file.read()).hexdigest()
        self.file.seek(0)
        return Image.query.filter(Image.md5_hash == self.md5_hash).first()
