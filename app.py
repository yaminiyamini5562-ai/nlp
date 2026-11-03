import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
DEFAULT_DATA = ROOT / "data" / "sample_feedback.csv"

st.set_page_config(page_title="EduFeedback AI", page_icon="🎓", layout="wide")
st.title("🎓 EduFeedback AI")
st.caption("NLP-based College Student Feedback Analysis")
st.write("Analyze student comments, identify common concerns, and explore sentiment by department and feedback area.")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

ASPECT_KEYWORDS = {
    "Teaching & Faculty": ["teacher", "faculty", "professor", "explain", "teaching", "lecture", "class"],
    "Laboratory": ["lab", "laboratory", "computer", "equipment", "systems", "experiment"],
    "Infrastructure": ["building", "classroom", "bench", "building", "wifi", "internet", "clean", "toilet", "water"],
    "Examinations": ["exam", "examination", "marks", "question paper", "assessment", "result"],
    "Administration": ["office", "administration", "certificate", "fee", "staff", "process"],
    "Timetable & Workload": ["timetable", "schedule", "period", "workload", "assignment", "deadline"]
}

def detect_aspects(text):
    low = str(text).lower()
    matches = [aspect for aspect, words in ASPECT_KEYWORDS.items()
               if any(word in low for word in words)]
    return ", ".join(matches) if matches else "General"

def rule_sentiment(text):
    # Fallback for the demo only. Use the trained model when available.
    low = str(text).lower()
    positive = ["good", "excellent", "helpful", "clear", "great", "well", "supportive", "clean", "improved", "love"]
    negative = ["poor", "bad", "late", "insufficient", "unclear", "difficult", "broken", "issue", "problem", "overcrowded", "not enough", "terrible"]
    p = sum(w in low for w in positive)
    n = sum(w in low for w in negative)
    if p > n:
        return "Positive"
    if n > p:
        return "Negative"
    return "Neutral"

def analyze(df):
    df = df.copy()
    df["feedback_text"] = df["feedback_text"].fillna("").astype(str).str.strip()
    df = df[df["feedback_text"] != ""].copy()
    if df.empty:
        return df
    if model is not None:
        df["sentiment"] = model.predict(df["feedback_text"])
    else:
        df["sentiment"] = df["feedback_text"].apply(rule_sentiment)
    df["detected_aspects"] = df["feedback_text"].apply(detect_aspects)
    return df

uploaded = st.file_uploader("Upload student feedback CSV", type=["csv"])
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    raw = pd.read_csv(DEFAULT_DATA)
    st.info("Showing sample data. Upload your own CSV to analyze it.")

if "feedback_text" not in raw.columns:
    st.error("CSV must contain a 'feedback_text' column. Optional columns: department, semester, aspect.")
    st.stop()

results = analyze(raw)
if results.empty:
    st.warning("No non-empty feedback comments found.")
    st.stop()

for col in ["department", "semester", "aspect"]:
    if col not in results.columns:
        results[col] = "Not specified"
    results[col] = results[col].fillna("Not specified").astype(str)

with st.sidebar:
    st.header("Filters")
    departments = sorted(results["department"].unique())
    chosen_departments = st.multiselect("Department", departments, default=departments)
    sentiments = sorted(results["sentiment"].unique())
    chosen_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
    aspects = sorted(set(a.strip() for values in results["detected_aspects"] for a in values.split(",")))
    chosen_aspects = st.multiselect("Detected aspect", aspects, default=aspects)

filtered = results[
    results["department"].isin(chosen_departments)
    & results["sentiment"].isin(chosen_sentiments)
    & results["detected_aspects"].apply(lambda value: any(a.strip() in chosen_aspects for a in value.split(",")))
]

if filtered.empty:
    st.warning("No feedback matches these filters.")
    st.stop()

a, b, c, d = st.columns(4)
a.metric("Feedback analyzed", len(filtered))
b.metric("Positive", int((filtered["sentiment"] == "Positive").sum()))
c.metric("Neutral", int((filtered["sentiment"] == "Neutral").sum()))
d.metric("Negative", int((filtered["sentiment"] == "Negative").sum()))

left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    sentiment_counts = filtered["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
    st.plotly_chart(px.pie(sentiment_counts, names="Sentiment", values="Count", hole=0.45), use_container_width=True)
with right:
    st.subheader("Feedback by department")
    dept_counts = filtered.groupby(["department", "sentiment"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(dept_counts, x="department", y="Count", color="sentiment", barmode="group"), use_container_width=True)

st.subheader("Common feedback areas")
aspect_rows = []
for _, row in filtered.iterrows():
    for aspect in row["detected_aspects"].split(","):
        aspect_rows.append({"Aspect": aspect.strip(), "Sentiment": row["sentiment"]})
if aspect_rows:
    aspect_df = pd.DataFrame(aspect_rows)
    aspect_counts = aspect_df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(aspect_counts, x="Aspect", y="Count", color="Sentiment", barmode="group"), use_container_width=True)

st.subheader("Analyzed feedback")
display_cols = [c for c in ["feedback_id", "department", "semester", "feedback_text", "sentiment", "detected_aspects"] if c in filtered.columns]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download analyzed feedback (CSV)", data=csv_bytes,
                   file_name="edufeedback_analysis.csv", mime="text/csv")

st.caption("Privacy note: use anonymized feedback and obtain appropriate permission before analyzing real student responses. Predictions are estimates and should not be used to penalize individual students or staff.")
