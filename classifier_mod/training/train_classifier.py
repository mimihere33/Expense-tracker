# ml/train_classifier.py
import os, datetime, json
import joblib
import pandas as pd
import unicodedata
import shutil
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from classifier_mod.training.pipeline import ExpensePipeline
from extensions import db
from models import User
from sqlalchemy import select
from app import app


MIN_LABELS = 200  # minimum labeled rows to prefer DB-only training

def load_db_labeled():
    rows = []
    users = User.query.all()
    for u in users:
        p = ExpensePipeline(db.session, u.id)
        p.load_raw()
        p.clean()
        df = p.get_for_nlp()
        # keep only confirmed labels (strong signal)
        if 'confirmed' in df.columns:
            df = df[df.get('confirmed', True) == True]
        # fallback: also accept non-uncategorized but unconfirmed (optional)
        df = df[df['category'].notna() & (df['category'] != 'uncategorized')]
        if len(df) > 0:
            rows.append(df[['description','category']])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=['description','category'])

def load_external_seed(path):
    # Load with fallback encodings to avoid foreign characters
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except:
        df = pd.read_csv(path, encoding="latin1")

    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]
    df.rename(columns={
        "description": "description",
        "category": "category",
        "sub_category": "subcategory"
    }, inplace=True)

    # Remove rows with missing description/category
    df = df.dropna(subset=["description", "category"])

    # Thorough text normalization
    df["description"] = df["description"].astype(str)

    # Fix encoding problems, strip accents
    df["description"] = df["description"].apply(
        lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    )

    # Lowercase
    df["description"] = df["description"].str.lower()

    # Remove weird symbols
    df["description"] = df["description"].str.replace(r"[^a-z0-9\s]", " ", regex=True)

    # Normalize spacing
    df["description"] = df["description"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Final safety clean
    df = df[df["description"].str.len() > 0]

    return df

def train_and_save(df_all):
    df_all = df_all.groupby("category").filter(lambda x: len(x) > 1)
    X = df_all['description'].astype(str).tolist()
    y = df_all['category'].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y if len(set(y))>1 else None
    )

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga'))
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # Save model + metadata
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('mod', exist_ok=True)
    model_path = f"mod/classifier_v{ts}.joblib"
    joblib.dump(pipe, model_path)

    meta = {
        "model": model_path,
        "timestamp": ts,
        "train_samples": len(df_all),
        "accuracy": acc,
        "class_report": report
    }
    with open(f"mod/classifier_meta_v{ts}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # update latest symlink (atomic)
    latest = 'mod/classifier_latest.joblib'
    try:
        import shutil
        shutil.copyfile(model_path, latest)
        print("Updated latest model copy:", latest)
    except Exception as e:
        print("Failed to update latest model:", e)
    return model_path, meta

def main():
    df_db = load_db_labeled()
    print("DB labeled rows:", len(df_db))
    if len(df_db) < MIN_LABELS:
        # fallback to external seed data + DB labels
        print("Not enough DB labels; loading external seed dataset")
        df_seed = load_external_seed("mod/data/transactions.csv")
        print("External seed rows:", len(df_seed))
        df_all = pd.concat([df_db, df_seed], ignore_index=True)
    else:
        df_all = df_db

    if df_all.empty:
        print("No labeled data available. Add some labeled expenses or provide seed CSV at mod/data/seed_transactions.csv")
        return

    train_and_save(df_all)

if __name__ == '__main__':
    with app.app_context():
        main()

