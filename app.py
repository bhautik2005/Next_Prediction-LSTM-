from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and assets
model = load_model("model/lstm_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/max_len_X.pkl", "rb") as f:
    max_len = pickle.load(f)

# Reverse tokenizer dictionary
index_to_word = {index: word for word, index in tokenizer.word_index.items()}


def predict_next_words(seed_text, num_words):
    output_text = seed_text

    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([output_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='pre')

        predicted = np.argmax(model.predict(sequence), axis=-1)[0]

        predicted_word = index_to_word.get(predicted, "")
        output_text += " " + predicted_word

    return output_text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    seed_text = data.get("text")
    num_words = int(data.get("num_words"))

    result = predict_next_words(seed_text, num_words)
    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
