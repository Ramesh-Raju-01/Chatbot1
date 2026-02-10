from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCasualLM

app = Flask(__name__)

# Load BlenderBot model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCasualLM.from_pretrained(model_name)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    reply_ids = model.generate(inputs,max_length=50, do_sample=True)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
