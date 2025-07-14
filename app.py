from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import joblib
from googlesearch import search

model = joblib.load("fake_news_model.pkl")

app = Flask(__name__)

def get_fact_check_links(query):
    try:
        results = list(search(query, num_results=3))
        return results
    except:
        return []

@app.route("/bot", methods=["POST"])
def bot():
    incoming_msg = request.values.get("Body", "").strip()
    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Please send a news headline or content.")
        return str(resp)

    prediction = model.predict([incoming_msg])[0]
    label = "REAL" if prediction == 1 else "FAKE"

    msg.body(f"This News is '{label}' \n ")

    if prediction == 0:
        links = get_fact_check_links(incoming_msg)
        if links:
            msg.body("\n Possible real news can be:\n" + "\n \n".join(links))
        else:
            msg.body("Couldn't find verified sources.")
    
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
