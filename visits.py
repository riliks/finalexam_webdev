import io
import math
import datetime
from datetime import datetime,timedelta
from flask import Blueprint, render_template, request, send_file,flash
from flask_login import current_user,login_required
from models import Book, User,Logs

bp = Blueprint('visits', __name__, url_prefix='/visits')

def convert_to_csv(fields):
    result= 'No,'
    for field in fields:
        result = result + ','+field
    result = result + '\n'
    # for i, record in enumerate(records):
    #     result += f"{i + 1}," + ','.join([str(getattr(record, j, '')) for j in fields]) + '\n'
    return result


def generate_report_file(records):
    buffer = io.BytesIO()
    buffer.write('Hello'.encode('utf-8'))
    # buffer.write(convert_to_csv(records).encode('utf-8'))
    buffer.seek(0)
    return buffer


@bp.route('/book_stat')
@login_required
def book_stat():
    if current_user.role==1:
        page = request.args.get('page', 1, type=int)
        log = Logs.query.order_by(Logs.created_at.desc())
        users=[]
        books=[]
        pg=page
        if page>1:
            pg=(page-1)*10
        for i in range(pg-1,page*10):
            try:
                user=User.query.filter_by(id=log[i].user_id).first()
                book=Book.query.filter_by(id=log[i].book_id).first()
                users.append(user.full_name)
                books.append(book.name)
            except:
                book=Book.query.filter_by(id=log[i].book_id).first()
                users.append('Неаутентифицированный пользователь')
                books.append(book.name)
        pagination = log.paginate(page, 10)
        log = pagination.items
        
        if request.args.get('download_csv'):
            f = generate_report_file(books)
            filename = datetime.now().strftime('%d_%m_%Y') + '_books_stat.csv'
            #return send_file(f, mimetype='text/csv', as_attachment=True, attachment_filename=filename)
        return render_template('visits/book_stat.html',pagination=pagination,records=log,users=users,books=books,pages=len(log))

@bp.route('/user_stat', methods=['GET','POST'])
@login_required
def user_stat():
    if current_user.role==1:
        fromdate = datetime.today() - timedelta(days=90)
        todate = fromdate + timedelta(days=90)
        if request.method == "POST":
            if request.form.get('fromdate'):
                fromdate=request.form.get('fromdate')
            if request.form.get('todate'):
                todate=request.form.get('todate')
        page = request.args.get('page', 1, type=int)
        books = Book.query.order_by(Book.view.desc())
        for book in books:
            amount=Logs.query.filter(Logs.book_id == book.id).filter(Logs.created_at >= fromdate).filter(Logs.created_at <= todate).count()
            book.view=amount
        pagination = books.paginate(page, 10)
        books = pagination.items
        flash(Book.query.all()[0].__table__._columns)
        if request.args.get('download_csv'):
            var=[]
            var.append('id')
            var.append('name')
            var.append('count')
            f = generate_report_file(var)
            filename =datetime.now().strftime('%d_%m_%Y') +'_user_stat.csv'
            #return send_file(f, mimetype='text/csv', as_attachment=True, attachment_filename=filename)
        return render_template('visits/user_stat.html',pagination=pagination,books=books,fromdate=request.form.get('fromdate'),todate=request.form.get('todate'))

    
    
    

if __name__ == '__main__':
    app.run()

