import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake['label'] = 0
df_true['label'] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)

X = df['title']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', PassiveAggressiveClassifier(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, 'fake_news_model.pkl')
print("Model saved as fake_news_model.pkl")
