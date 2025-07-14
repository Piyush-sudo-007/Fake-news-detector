import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake['label'] = "FAKE"
df_true['label'] = "REAL"

df = pd.concat([df_fake, df_true], ignore_index=True)

X = df['text']
y = df['label'].map({'REAL': 1, 'FAKE': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

model.fit(X_train, y_train)

joblib.dump(model, 'fake_news_model.pkl')
print("Model saved as fake_news_model.pkl")
