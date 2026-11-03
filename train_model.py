from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_feedback.csv"
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"

df = pd.read_csv(DATA_PATH)
required = {"feedback_text", "sentiment"}
if not required.issubset(df.columns):
    raise ValueError("Training CSV must contain feedback_text and sentiment columns.")

df = df.dropna(subset=["feedback_text", "sentiment"]).copy()
df["feedback_text"] = df["feedback_text"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.title()
df = df[df["feedback_text"] != ""]
df = df[df["sentiment"].isin(["Positive", "Neutral", "Negative"])]

counts = df["sentiment"].value_counts()
if len(counts) < 3 or counts.min() < 2:
    raise ValueError("Need at least 2 examples for each of Positive, Neutral, and Negative.")

X_train, X_test, y_train, y_test = train_test_split(
    df["feedback_text"], df["sentiment"],
    test_size=0.25, random_state=42, stratify=df["sentiment"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, zero_division=0))
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
