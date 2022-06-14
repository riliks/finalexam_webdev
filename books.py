from ast import Try
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user

from app import db
from models import Category, Book, User, Review, Image
from tools import BooksFilter, ImageSaver

bp = Blueprint('books', __name__, url_prefix='/books')

BOOKS_PARAMS = ['name','category_id', 'author' ,'pub_house',  'short_desc', 'volume']
PER_PAGE = 5

def params():
    return { p: request.form.get(p) for p in BOOKS_PARAMS }


def search_params():
    return {
        'name': request.args.get('name'),
        'category_ids': request.args.getlist('category_ids')
    }

@bp.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    books = BooksFilter(**search_params()).perform()
    pagination = books.paginate(page, PER_PAGE)
    categories = Category.query.all()
    return render_template('books/index.html', categories=categories, search_params=search_params(),user=current_user,books=books,pagination=pagination)


@bp.route('/new')
@login_required
def new():
    categories = Category.query.all()
    users = User.query.all()
    return render_template('books/new.html', categories=categories, users=users)


@bp.route('/create', methods=['GET','POST'])
@login_required
def create():

    f = request.files.get('background_img')
    if f and f.filename:
        img = ImageSaver(f).save()

    book = Book(**params(), image_id=img.id)
    try:
        db.session.add(book)
        db.session.commit()
    except:
        db.session.rollback()

    flash(f'Курс {book.name} был успешно добавлен!', 'success')

    return redirect(url_for('books.index'))

@bp.route('/<int:book_id>/review', methods=['POST'])
@login_required
def review(book_id):
    review = Review(rating = request.form.get('mark'),text = request.form.get('review'), book_id=book_id,user_id=request.form.get('user_id'))
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
@login_required
def show(book_id,):
    book = Book.query.get(book_id)
    reviews = Review.query.order_by(Review.created_at.desc()).filter(Review.book_id == book_id)
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
@login_required
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
     
    return render_template('books/reviewlist.html', reviews=reviews, f=filter, pagination=pagination)

@bp.route('/delete', methods=['POST'])
@login_required
def delete():
    # obj = Book.query.filter_by(id=request.form.get('book_id')).one()
    # db.session.delete(obj)
    # db.session.commit()
    flash(request.form.get('book_id'))
    flash('Вами была удалена книга: '+request.form.get('book_name'),'danger')
    page = request.args.get('page', 1, type=int)
    books = BooksFilter(**search_params()).perform()
    pagination = books.paginate(page, PER_PAGE)
    categories = Category.query.all()
    return render_template('books/index.html', categories=categories, search_params=search_params(),user=current_user,books=books,pagination=pagination)

@bp.route('/edit', methods=['POST'])
@login_required
def edit():
    # obj = Book.query.filter_by(id=request.form.get('book_id')).one()
    # db.session.delete(obj)
    # db.session.commit()
    flash('Книга '+request.form.get('book_name')+' была успешна обновлена!','success')
    page = request.args.get('page', 1, type=int)
    books = BooksFilter(**search_params()).perform()
    pagination = books.paginate(page, PER_PAGE)
    categories = Category.query.all()
    return render_template('books/index.html', categories=categories, search_params=search_params(),user=current_user,books=books,pagination=pagination)
