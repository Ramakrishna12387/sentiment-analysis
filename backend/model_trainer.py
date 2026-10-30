import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# மாதிரி விமர்சனத் தரவுகள் (Sample Dataset)
data = {
    "review": [
        "The product quality is amazing, loved it!",
        "Delivery was delayed by two weeks, terrible service.",
        "Battery life is fantastic and screen is vibrant.",
        "Worst packaging ever, item arrived broken and damaged.",
        "Customer support was very helpful and solved my issue.",
        "Poor quality material, stopped working after two days.",
        "Value for money, totally recommended!",
        "Late shipping and bad customer response.",
    ],
    "sentiment": [
        "Positive",
        "Negative",
        "Positive",
        "Negative",
        "Positive",
        "Negative",
        "Positive",
        "Negative",
    ],
}

df = pd.DataFrame(data)

# 1. Sentiment Classifier Pipeline (TF-IDF + Logistic Regression)
sentiment_pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression()),
    ]
)
sentiment_pipeline.fit(df["review"], df["sentiment"])

# 2. Topic Modeling Pipeline (LDA)
count_vectorizer = CountVectorizer(stop_words="english", max_features=50)
count_matrix = count_vectorizer.fit_transform(df["review"])

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(count_matrix)

# மாடல்களைச் சேமித்தல்
joblib.dump(sentiment_pipeline, "sentiment_model.pkl")
joblib.dump((count_vectorizer, lda), "topic_model.pkl")
print("Models trained and saved successfully!")