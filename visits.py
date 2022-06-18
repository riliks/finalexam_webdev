import io
import math
import datetime
from flask import Blueprint, render_template, request, send_file
from flask_login import current_user

bp = Blueprint('visits', __name__, url_prefix='/visits')

PER_PAGE = 5


def convert_to_csv(records):
    fields = records[0]._fields
    result = 'No,' + ','.join(fields) + '\n'
    for i, record in enumerate(records):
        result += f"{i + 1}," + ','.join([str(getattr(record, j, '')) for j in fields]) + '\n'
    return result


def generate_report_file(records):
    buffer = io.BytesIO()
    buffer.write(convert_to_csv(records).encode('utf-8'))
    buffer.seek(0)
    return buffer


@bp.route('/book_stat')
def book_stat():
    return render_template('visits/book_stat.html')

@bp.route('/user_stat')
def user_stat():
    return render_template('visits/user_stat.html')

if __name__ == '__main__':
    app.run()

