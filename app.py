import numpy
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pre import FeatureExtractor
# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/predict", methods = ["POST"])
def predict():
    jar=request.json
    query_df=pd.DataFrame(jar)
    d=FeatureExtractor(query_df)
    prediction = list(model.predict(d.transforms()))
    return jsonify({'Predicted - >':prediction})

if __name__ == "__main__":
    flask_app.run(debug=True)