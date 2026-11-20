import itertools
import os
import random

import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

random.seed(42)

# ---------------------------------------------------------------
# 1. Dataset
#    a) Product-review sentences generated from many phrase
#       combinations (quality, delivery, packaging, service, price,
#       battery/display).
#    b) Optional: உங்கள் சொந்த real reviews இருந்தால் backend/reviews.csv
#       (columns: review,sentiment) என்று வைத்தால் தானாக சேர்க்கப்படும்.
# ---------------------------------------------------------------
ASPECTS = {
    "quality": {
        "Positive": [
            "The product quality is amazing",
            "Build quality is excellent and feels premium",
            "Very sturdy and well made",
            "The material is top notch and durable",
            "Works perfectly, no issues at all",
            "Great finish and solid construction",
        ],
        "Negative": [
            "The product quality is terrible",
            "Poor quality material, feels cheap",
            "Stopped working after two days",
            "Build is flimsy and broke very quickly",
            "Defective piece, does not work properly",
            "Scratches all over and looks used",
        ],
    },
    "delivery": {
        "Positive": [
            "Delivery was super fast",
            "Arrived a day before the expected date",
            "Shipping was quick and the tracking was accurate",
            "The courier was polite and delivered on time",
            "Got it within two days, very prompt delivery",
        ],
        "Negative": [
            "Delivery was delayed by two weeks",
            "Late shipping and no tracking updates",
            "The courier never showed up on the promised date",
            "Package took forever to arrive",
            "Delivered to the wrong address and nobody helped",
        ],
    },
    "packaging": {
        "Positive": [
            "Packaging was neat and secure",
            "Everything was wrapped carefully with bubble wrap",
            "The box was sealed properly and the item was intact",
            "Beautiful packaging, arrived in perfect condition",
        ],
        "Negative": [
            "Worst packaging ever, item arrived broken",
            "The box was crushed and the item was damaged",
            "Packaging was torn and the seal was open",
            "No protection inside, everything arrived damaged",
        ],
    },
    "service": {
        "Positive": [
            "Customer support was very helpful and solved my issue",
            "The service team replied quickly and was very polite",
            "Got a replacement within 24 hours without any hassle",
            "Excellent after sales support",
            "Refund was processed smoothly and quickly",
        ],
        "Negative": [
            "Customer support was rude and unhelpful",
            "Nobody replied to my emails for a week",
            "Bad customer response and no solution offered",
            "They refused to give a refund or replacement",
            "Support kept transferring my call and never fixed anything",
        ],
    },
    "price": {
        "Positive": [
            "Totally value for money",
            "Great price for what you get",
            "Worth every rupee",
            "Cheaper than other shops and the same quality",
        ],
        "Negative": [
            "Overpriced for such low quality",
            "Not worth the money at all",
            "Waste of money, cheaper options are much better",
            "Way too expensive for what it offers",
        ],
    },
    "features": {
        "Positive": [
            "Battery life is fantastic and the screen is vibrant",
            "The display is incredibly sharp and the sound is clear",
            "Super smooth performance and a very fast charging speed",
            "The camera quality is brilliant",
        ],
        "Negative": [
            "Battery drains very fast and the screen flickers",
            "The display is dull and the sound is distorted",
            "Very laggy performance and slow charging",
            "The camera quality is awful",
        ],
    },
}

CLOSERS = {
    "Positive": [
        "",
        "Totally recommended!",
        "Loved it.",
        "Will buy again.",
        "Very happy with this purchase.",
        "Five stars.",
        "Highly satisfied.",
    ],
    "Negative": [
        "",
        "Not recommended.",
        "Very disappointed.",
        "Will never buy again.",
        "Complete waste.",
        "One star.",
        "Highly dissatisfied.",
    ],
}


def build_synthetic_dataset(max_per_class=450):
    rows = []
    for sentiment in ("Positive", "Negative"):
        phrases = [p for a in ASPECTS.values() for p in a[sentiment]]
        combos = []
        # single phrase + closer
        for p, c in itertools.product(phrases, CLOSERS[sentiment]):
            combos.append(f"{p}. {c}".strip())
        # two-phrase reviews, more realistic
        for p1, p2 in itertools.permutations(phrases, 2):
            c = random.choice(CLOSERS[sentiment])
            combos.append(f"{p1}, and {p2[0].lower() + p2[1:]}. {c}".strip())
        random.shuffle(combos)
        for text in combos[:max_per_class]:
            rows.append({"review": text, "sentiment": sentiment})
    return pd.DataFrame(rows)


df = build_synthetic_dataset()

if os.path.exists("reviews.csv"):
    real = pd.read_csv("reviews.csv")[["review", "sentiment"]].dropna()
    df = pd.concat([df, real], ignore_index=True)
    print(f"Added {len(real)} real reviews from reviews.csv")

df = df.drop_duplicates("review").reset_index(drop=True)
print(f"Total reviews: {len(df)}")
print(df["sentiment"].value_counts().to_string())

# ---------------------------------------------------------------
# 2. Sentiment Classifier (TF-IDF + Logistic Regression)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

sentiment_pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5)),
    ]
)
sentiment_pipeline.fit(X_train, y_train)

pred = sentiment_pipeline.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred))

# Final model: முழு dataset-ல் மீண்டும் train
sentiment_pipeline.fit(df["review"], df["sentiment"])

# ---------------------------------------------------------------
# 3. Topic Modeling (LDA, 3 topics)
# ---------------------------------------------------------------
count_vectorizer = CountVectorizer(stop_words="english", max_features=300, min_df=3)
count_matrix = count_vectorizer.fit_transform(df["review"])

lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=30)
lda.fit(count_matrix)

feature_names = count_vectorizer.get_feature_names_out()
print("\nDiscovered topics:")
for i, comp in enumerate(lda.components_):
    top = [feature_names[j] for j in comp.argsort()[:-6:-1]]
    print(f"  Topic {i}: {', '.join(top)}")

# ---------------------------------------------------------------
# 4. Save models
# ---------------------------------------------------------------
joblib.dump(sentiment_pipeline, "sentiment_model.pkl")
joblib.dump((count_vectorizer, lda), "topic_model.pkl")
print("\nModels trained and saved successfully!")
