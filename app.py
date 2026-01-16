import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from sqlalchemy import extract, func
from dotenv import load_dotenv
import plotly.express as px
import plotly
import json
import pandas as pd
import plotly.graph_objs as go
import joblib

# Load environment variables
if os.environ.get("RENDER") is None:
    load_dotenv()

from extensions import db
from models import Expense, User 

app = Flask(__name__)

# Security: Use environment variables
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///expenses.db')

# Fix for Render PostgreSQL URL format
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 300,
    'pool_pre_ping': True,
}

# Security headers (enable in production)
if os.environ.get('RENDER'):
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML classifier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mod", "classifier_latest.joblib")
classifier = None
try:
    classifier = joblib.load(MODEL_PATH)
    print(f"✓ Loaded classifier: {MODEL_PATH}")
except Exception as e:
    print(f"⚠ Classifier not loaded: {e}")

# Create tables if they don't exist
with app.app_context():
    db.create_all()


# ---------------- AUTH ROUTES ---------------- #

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(username=username).first():
            flash('Username already taken!', 'error')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_pw)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Signup successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')
            print(f"Signup error: {e}")

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password required!', 'error')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

        login_user(user)
        flash(f'Welcome back, {user.username}!', 'success')
        
        # Redirect to next page if exists
        next_page = request.args.get('next')
        return redirect(next_page) if next_page else redirect(url_for('home'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ---------------- MAIN APP ROUTES ---------------- #

@app.route('/')
@login_required
def home():
    expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date.desc()).limit(5).all()
    total_expenses = db.session.query(func.sum(Expense.amount)).filter(Expense.user_id == current_user.id).scalar() or 0
    return render_template('home.html', expenses=expenses, total=total_expenses)


@app.route('/add', methods=['POST'])
@login_required
def add_exp():
    amount = request.form.get('amount')
    
    # Validate amount
    try:
        amount = float(amount)
        if not (0 < amount < 1e8):
            raise ValueError("Amount out of range")
    except:
        flash("Invalid amount. Please enter a valid number.", "error")
        return redirect(url_for('log'))

    note = request.form.get('note', '').strip()
    if len(note) > 500:
        note = note[:500]

    predicted_cat = None
    predicted_conf = None
    user_category = request.form.get('category', '').strip()

    # ML prediction
    if classifier is not None and note:
        try:
            pred = classifier.predict([note])[0]
            prob = classifier.predict_proba([note])[0]
            conf = max(prob)

            predicted_cat = pred
            predicted_conf = float(conf)
            
            # Use prediction if user didn't select category
            if not user_category:
                user_category = predicted_cat
        except Exception as e:
            print(f"Prediction error: {e}")

    # Ensure category exists
    if not user_category:
        user_category = "Other"

    new_expense = Expense(
        user_id=current_user.id,
        note=note,
        amount=amount,
        category=user_category,
        predicted_category=predicted_cat,
        predicted_confidence=predicted_conf,
        confirmed=False,
        date=datetime.now()
    )

    try:
        db.session.add(new_expense)
        db.session.commit()
        flash(f"✓ Expense added: ₹{amount:.2f} in {user_category}", "success")
    except Exception as e:
        db.session.rollback()
        flash("Error adding expense. Please try again.", "error")
        print(f"Add expense error: {e}")
    
    return redirect(url_for('log'))


@app.route('/delete/<int:id>')
@login_required
def del_exp(id):
    expense = Expense.query.get_or_404(id)
    
    if expense.user_id != current_user.id:
        flash("You can only delete your own expenses.", "error")
        return redirect(url_for('log'))
    
    try:
        db.session.delete(expense)
        db.session.commit()
        flash("Expense deleted successfully.", "info")
    except Exception as e:
        db.session.rollback()
        flash("Error deleting expense.", "error")
        print(f"Delete error: {e}")
    
    return redirect(url_for('log'))


@app.route('/update/<int:id>', methods=['POST'])
@login_required
def update_exp(id):
    expense = Expense.query.get_or_404(id)
    
    if expense.user_id != current_user.id:
        flash("Unauthorized action.", "error")
        return redirect(url_for('log'))

    try:
        expense.amount = float(request.form.get('amount', expense.amount))
        expense.category = request.form.get('category', expense.category).strip()
        expense.note = request.form.get('note', expense.note).strip()
        
        db.session.commit()
        flash("Expense updated successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash("Error updating expense.", "error")
        print(f"Update error: {e}")
    
    return redirect(url_for('log'))


@app.route('/log')
@login_required
def log():
    expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date.desc()).all()

    # Get user's existing categories
    user_categories = db.session.query(Expense.category).filter(
        Expense.user_id == current_user.id,
        Expense.category.isnot(None)
    ).distinct().all()
    
    categories = sorted(set([c[0] for c in user_categories if c[0]]))
    
    # Add default categories if not present
    default_cats = ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Health", "Transport", "Other"]
    for cat in default_cats:
        if cat not in categories:
            categories.append(cat)

    return render_template("log.html", expenses=expenses, categories=categories)


@app.route('/analysis')
@login_required
def analysis():
    # Category-wise spending
    category_sums = db.session.query(
        Expense.category, func.sum(Expense.amount)
    ).filter(Expense.user_id == current_user.id).group_by(Expense.category).all()

    pie_labels = [c[0] or "Uncategorized" for c in category_sums]
    pie_values = [float(c[1]) for c in category_sums]

    # Daily spending trend
    daily_data = db.session.query(
        Expense.category,
        func.date(Expense.date),
        func.sum(Expense.amount)
    ).filter(Expense.user_id == current_user.id).group_by(
        Expense.category, func.date(Expense.date)
    ).all()

    daily_df = pd.DataFrame(daily_data, columns=["category", "date", "amount"])

    # Time-based totals
    today = datetime.now().date()
    month = datetime.now().month
    year = datetime.now().year

    today_total = db.session.query(func.sum(Expense.amount))\
        .filter(Expense.user_id == current_user.id, func.date(Expense.date) == today).scalar() or 0

    month_total = db.session.query(func.sum(Expense.amount))\
        .filter(Expense.user_id == current_user.id,
                extract('month', Expense.date) == month,
                extract('year', Expense.date) == year).scalar() or 0

    year_total = db.session.query(func.sum(Expense.amount))\
        .filter(Expense.user_id == current_user.id,
                extract('year', Expense.date) == year).scalar() or 0

    overall_total = sum(pie_values)
    percentages = {label: round((val / overall_total) * 100, 2) if overall_total > 0 else 0
                   for label, val in zip(pie_labels, pie_values)}

    # Create charts
    pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values)])
    bar_df = pd.DataFrame({"Category": pie_labels, "Total": pie_values})
    bar = px.bar(bar_df, x="Category", y="Total", title="Total by Category")
    line = px.line(daily_df, x="date", y="amount", color="category",
                   title="Daily Spending by Category")

    pie_json = json.dumps(pie, cls=plotly.utils.PlotlyJSONEncoder)
    bar_json = json.dumps(bar, cls=plotly.utils.PlotlyJSONEncoder)
    line_json = json.dumps(line, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "analysis.html",
        pie_json=pie_json,
        bar_json=bar_json,
        line_json=line_json,
        today_total=today_total,
        month_total=month_total,
        year_total=year_total,
        overall_total=overall_total,
        percentages=percentages
    )


@app.route('/predict_category', methods=['POST'])
@login_required
def predict_category():
    note = request.form.get('note', '').strip()
    
    if not note or classifier is None:
        return jsonify({'category': None, 'confidence': None})

    try:
        prob = classifier.predict_proba([note])[0]
        pred = classifier.predict([note])[0]
        conf = max(prob)
        
        return jsonify({
            'category': pred,
            'confidence': float(conf)
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'category': None, 'confidence': None, 'error': str(e)})


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return "Not Found", 404

@app.errorhandler(500)
def server_error(e):
    return "Not Found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
