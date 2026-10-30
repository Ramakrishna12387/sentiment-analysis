import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sentiment & Topic Analysis API")

# Netlify-லிருந்து வரும் கோரிக்கைகளை அனுமதிக்க CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# மாடல்களை ஏற்றுதல்
sentiment_model = joblib.load("sentiment_model.pkl")
count_vectorizer, lda_model = joblib.load("topic_model.pkl")


class ReviewInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "NLP Backend is running"}


@app.post("/predict")
def predict_sentiment_and_topic(review: ReviewInput):
    # 1. Sentiment கணிப்பு
    prediction = sentiment_model.predict([review.text])[0]
    probabilities = sentiment_model.predict_proba([review.text])[0]
    confidence = max(probabilities) * 100

    # 2. Topic Modeling (முக்கிய வார்த்தைகளை பிரித்தெடுத்தல்)
    vec = count_vectorizer.transform([review.text])
    topic_distribution = lda_model.transform(vec)[0]

    feature_names = count_vectorizer.get_feature_names_out()
    primary_topic_idx = topic_distribution.argmax()
    top_words_idx = lda_model.components_[primary_topic_idx].argsort()[:-4:-1]
    topic_keywords = [feature_names[i] for i in top_words_idx]

    return {
        "sentiment": prediction,
        "confidence": f"{confidence:.2f}%",
        "primary_topic_id": int(primary_topic_idx),
        "topic_keywords": topic_keywords,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)