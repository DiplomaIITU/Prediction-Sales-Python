import numpy
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pre import FeatureExtractor
# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
import json
@flask_app.route("/predict", methods = ["POST"])
def predict():
    jar=request.json
    query_df=pd.DataFrame(jar)
    d=FeatureExtractor(query_df)
    prediction = list(model.predict(d.transforms()))
    lest=[int(v) for dct in jar for k,v in dct.items()]
    lest=lest[2::3]
    valuess=dict(zip(lest,prediction))
    return jsonify(valuess)

if __name__ == "__main__":
    flask_app.run(debug=True)