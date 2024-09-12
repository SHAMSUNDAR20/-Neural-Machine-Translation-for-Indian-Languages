import torch
from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, request, jsonify

# Initialize the model and tokenizer for translation (e.g., Hindi to English)
model_name = 'Helsinki-NLP/opus-mt-hi-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

# Translation function
def translate_text(text, src_lang='hi', tgt_lang='en'):
    translated = model.generate(**tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# API for translation
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    input_text = data.get('text')
    output = translate_text(input_text)
    return jsonify({'translated_text': output})

if __name__ == "__main__":
    app.run(debug=True)
