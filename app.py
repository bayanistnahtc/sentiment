from flask import Flask, request #render_template
# from pydantic.main import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# app = FastAPI()
model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)


app = Flask(__name__)

# @app.route('predict')
@app.route('/predict', methods=['POST'])
def home():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        # print(json)
        # return json

        """ Calculate sentiment of a text. `return_type` can be 'label', 'score' or 'proba' """
        # with torch.no_grad():
        inputs = tokenizer(json['text'], return_tensors='pt', truncation=True, padding=True).to(model.device)
        # proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
        return str({
            2: 1,
            1: 3,
            0: 2
        }[int(model(**inputs).logits.detach().numpy().argmax())])
    else:
        return 'Content-Type not supported!'




# def home():
#     return render_template("index.html")

app.run(debug=True)
