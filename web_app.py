from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Neural Machine Translation for Indian Languages"

if __name__ == "__main__":
    app.run(debug=True)
