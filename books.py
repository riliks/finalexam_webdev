from ast import Try
from distutils.log import error
from unicodedata import category
from flask import Blueprint, render_template, request, flash, redirect, url_for, make_response, request
from flask_login import login_required, current_user
import os
from app import db
from models import Category, Book, User, Review, Image,Logs
from tools import ImageSaver,check_text
import bleach
from datetime import datetime, date, time,timedelta
bp = Blueprint('books', __name__, url_prefix='/books')

BOOKS_PARAMS = ['name','year', 'author' ,'pub_house',  'short_desc', 'volume']
PER_PAGE = 10

def params():
    return { p: request.form.get(p) for p in BOOKS_PARAMS }

def checkform():
    errors=[]
    for p in BOOKS_PARAMS:
        if p=='year':
            errors.append(check_text(request.form.get(p),'year'))
        else:
            errors.append(check_text(request.form.get(p))) # 0-5 params , 6 all, 7 category
    errors.append(False)
    errors.append(True)
    for i in range(0,100):
        if request.form.get(f"category{i}"):
            errors[7]=False
    for i in errors:
        if i:
            errors[6]=True
    return errors

def search_params():
    return {
        'name': request.args.get('name'),
        'category_ids': request.args.getlist('category_ids')
    }

@bp.route('/')
def index():
    looked=False
    if current_user.is_authenticated:
        looked=Logs.query.filter(Logs.user_id == current_user.id).count()>0
    else:
        looked=Logs.query.filter(Logs.user_id == None).count()>0
    page = request.args.get('page', 1, type=int)
    books = Book.query.order_by(Book.created_at.desc())
    pagination = books.paginate(page, PER_PAGE)
    categories = Category.query.all()
    return render_template('books/index.html', categories=categories, search_params=search_params(),user=current_user,books=books,pagination=pagination,looked=looked)


@bp.route('/new')
@login_required
def new():
    categories = Category.query.all()
    users = User.query.all()
    return render_template('books/new.html', categories=categories, users=users,errors='',book='')


@bp.route('/create', methods=['GET','POST'])
@login_required
def create():
    if current_user.role==1:
        errors=checkform()
        book =  Book(**params())
        var=[]
        for i in range(0,100):
            if request.form.get(f"category{i}"):
                var.append(Category.query.filter_by(id=i).one())
        book.categories=var
        if errors[6]:
            categories = Category.query.all()
            users = User.query.all()
            return render_template('books/new.html', categories=categories, users=users,errors=errors,book=book)
        else:
            f = request.files.get('background_img')
            if f and f.filename:
                img = ImageSaver(f).save()
            book.image_id=img.id
            book.short_desc=bleach.clean(book.short_desc)
            try:
                db.session.add(book)
                db.session.commit()
                flash(f'???????????????????????? {book.name} ???????? ?????????????? ??????????????????!', 'success')
            except:
                db.session.rollback()
            
            return redirect(url_for('books.index'))
    else:
        flash('???????????????????????? ????????','danger')

@bp.route('/<int:book_id>/review', methods=['POST'])
@login_required
def review(book_id):
    review = Review(rating = request.form.get('mark'),text = bleach.clean(request.form.get('review')), book_id=book_id,user_id=request.form.get('user_id'))
    book = Book.query.get(book_id)
    book.addmark(int(request.form.get('mark')))
    try:
        db.session.add(review)
        db.session.commit()
        db.session.add(book)
        db.session.commit()
    except:
        db.session.rollback()
    return redirect(url_for('books.show', book_id=book_id))

@bp.route('/<int:book_id>')
def show(book_id,):
    book = Book.query.get(book_id)
    reviews = Review.query.order_by(Review.created_at.desc()).filter(Review.book_id == book_id)
    date = datetime.today() - timedelta(days=1)
    if current_user.is_authenticated:
        if Logs.query.filter(Logs.user_id == current_user.id).filter(Logs.book_id == book_id).filter(Logs.created_at > date).count()<10:
            log = Logs(user_id=current_user.id,book_id=book_id)
            book.view=book.view+1
    else:
        if Logs.query.filter(Logs.user_id == None).filter(Logs.book_id == book_id).filter(Logs.created_at > date).count()<10:
            log = Logs(user_id=None,book_id=book_id)
    try:
        db.session.add(log)
        db.session.commit()
        db.session.add(book)
        db.session.commit()
    except:
        db.session.rollback()
    check=None
    for i in reviews:
        try:
            if i.user_id==current_user.id:
                check=i
        except:
            pass
        
    reviews = reviews.limit(5)
    return render_template('books/show.html', book=book, reviews=reviews, check=check)

@bp.route('/<int:book_id>/reviewlist', methods=['GET','POST'])
def reviewlist(book_id):
    page = request.args.get('page', 1, type=int)
    query =  Review.query.filter(Review.book_id == book_id)
    reviews = query.order_by(Review.created_at.desc())
    filter=request.form.get('filter') or request.args.get('f', type=str)
    if filter=='last':
        reviews =  query.order_by(Review.created_at.asc())
    elif filter=='better':
        reviews =  query.order_by(Review.rating.desc())
    elif filter=='worse':
        reviews =  query.order_by(Review.rating.asc())

    pagination = reviews.paginate(page,PER_PAGE)
    reviews = pagination.items
     
    return render_template('books/reviewlist.html', reviews=reviews,book_id=book_id, f=filter, pagination=pagination)

@bp.route('/delete', methods=['POST'])
@login_required
def delete():
    if current_user.role==1:
        obj = Book.query.filter_by(id=request.form.get('book_id')).one()
        obj2 = Image.query.filter_by(id=obj.image_id).one()
        try:
            db.session.delete(obj)
            db.session.commit()
            db.session.delete(obj2)
            db.session.commit()
            flash('???????? ???????? ?????????????? ??????????: '+request.form.get('book_name'),'danger')
            os.remove(f"/home/std/web_dev/Exam4/media/images/{obj.image_id}.jpg")
        except:
            flash('?????????????????? ???????????? ?????? ????????????????, ?????????????????? ??????????','danger')
            db.session.rollback()
        return redirect(url_for('books.index'))
    else:
        flash('???????????????????????? ????????','danger')
@bp.route('/edit', methods=['POST'])
@login_required
def edit():
    if current_user.role==1 or current_user.role==2:
        errors=checkform()
        if errors[6]:
            flash('?????????????????? ???????????? ?????? ??????????????????. ???????? ???? ?????????? ???????? ????????????','danger')
            return redirect(url_for('books.index'))
        else:
            obj = Book.query.filter_by(id=request.form.get('book_id')).one()
            p=BOOKS_PARAMS
            obj.name=request.form.get(p[0])
            obj.year=request.form.get(p[1])
            obj.author=request.form.get(p[2])
            obj.pub_house=request.form.get(p[3])
            obj.short_desc=bleach.clean(request.form.get(p[4]))
            obj.volume=request.form.get(p[5])      
            try:
                db.session.add(obj)
                db.session.commit()
                flash(f'?????????????????????? {obj.name} ???????? ?????????????? ????????????????!', 'success')
            except:
                db.session.rollback()
                flash('?????????????????? ???????????? ?????? ??????????????????, ?????????????????? ??????????','danger')
        return redirect(url_for('books.index'))
    else:
        flash('???????????????????????? ????????','danger')


@bp.route('/popular')
def popular():
    books = Book.query.order_by(Book.view.desc()).all()
    date = datetime.today() - timedelta(days=90)
    logs=Logs.query.all()
    for book in books:
        amount=Logs.query.filter(Logs.book_id == book.id).filter(Logs.created_at > date).count()
        book.view=amount
    return render_template('books/popular.html',books=books)

@bp.route('/viewed')
def viewed():
    if current_user.is_authenticated:
        log=Logs.query.filter(Logs.user_id == current_user.id).order_by(Logs.book_id.desc()).all()
    else:
        log=Logs.query.filter(Logs.user_id == None).order_by(Logs.book_id.desc()).all()
    list=[]
    list.append(log[0].book_id)
    for i in log:
        if list[len(list)-1]!=i.book_id:
            list.append(i.book_id)
    books = Book.query.order_by(Book.created_at.desc()).limit(5) 
    var=[] 
    for book in books:
        if book.id in list:var.append(book)
    return render_template('books/viewed.html',books=var)