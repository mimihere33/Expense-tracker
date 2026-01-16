import pandas as pd
import re
from sqlalchemy.orm import Session
from models import Expense  #SQLAlchemy model

class ExpensePipeline:
    def __init__(self, db_session = None, user_id = None):
        self.db = db_session
        self.user_id = user_id
        self.df = None

    # STEP 1: LOAD RAW EXPENSES FROM POSTGRES

    def load_raw(self):
        expenses = (
            self.db.query(Expense)
            .filter(Expense.user_id == self.user_id)
            .all()
        )
        self.df = pd.DataFrame([
            {
                "date": e.date,
                "description": e.description,
                "amount": e.amount,
                "category": e.category
            }
            for e in expenses
        ])
        return self.df

    # STEP 2: CLEAN THE DATA

    def clean(self):
        df = self.df.copy()

        # Clean date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")

        # Clean amount
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["amount"])
        
        # Clean description text
        df["description"] = df["description"].fillna("").astype(str)
        df["description"] = df["description"].str.lower()
        df["description"] = df["description"].apply(self._clean_text)

        # Normalize category text
        if "category" in df.columns:
            df["category"] = df["category"].fillna("uncategorized")
            df["category"] = df["category"].str.lower().str.strip()

        self.df = df
        return df

    # text cleaning
    def _clean_text(self, text):
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove symbols
        text = re.sub(r"\s+", " ", text)          # normalize spaces
        return text.strip()

    # OUTPUTS FOR SPECIFIC MODELS

    # For NLP classifier
    def get_for_nlp(self):
        return self.df[["description", "category"]]

    # For LSTM forecasting
    def get_for_timeseries(self):
        ts = self.df[["date", "amount"]].copy()
        ts = ts.set_index("date").sort_index()
        return ts

    # For insights/statistics
    def get_full_clean(self):
        return self.df
