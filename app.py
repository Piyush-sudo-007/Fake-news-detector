from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import joblib
from googlesearch import search
import os
import datetime

model = joblib.load("fake_news_model.pkl")

app = Flask(__name__)
MAX_LEN = 1000

def get_fact_check_links(query):
    try:
        results = list(search(f"{query} , is this news true?", num_results=3))
        return results
    except Exception as e:
        return []

def log_user(user, message, result):
    with open("query_logs.txt", "a") as f:
        timestamp = datetime.datetime.now()
        f.write(f"{timestamp} || {user} || {message[:100]} || {result}\n")
    
    if not os.path.exists("user_logs.txt"):
        open("user_logs.txt", "w").close()
    
    with open("user_logs.txt", "r") as f:
        existing_users = set(line.strip() for line in f)
    
    if user not in existing_users:
        with open("user_logs.txt", "a") as f:
            f.write(f"{user}\n")

def clean_input(text):
    return text.lower().strip()[:MAX_LEN]

@app.route("/bot", methods=["POST"])
def bot():
    incoming_msg = request.values.get("Body", "").strip()
    user_number = request.values.get("From", "")

    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Please send a news headline or article.")
        return str(resp)

    incoming_msg = clean_input(incoming_msg)

    if incoming_msg in ["hi", "hello", "hey"]:
        msg.body("Hi! I'm News Verifier Bot. Send me a news headline or short content to verify it.")
        return str(resp)

    try:
        prediction = model.predict([incoming_msg])[0]
        label = "REAL" if prediction == 1 else "FAKE"

        msg.body(f"This news appears to be '{label}'.\n")

        log_user(user_number, incoming_msg, label)

        
        links = get_fact_check_links(incoming_msg)
        if links:
            msg.body("For your reference:\n" + "\n\n".join(links))
        else:
            msg.body("Couldn't find verified sources.")

    except Exception as e:
        print(e)
        msg.body("Sorry, something went wrong while processing your message.")

    return str(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
