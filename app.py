
from modules.kmeans import kmeans
from modules.gauss import gauss
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os.path
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

save_path = '/uploads/'
exts = ['csv', 'json', 'yaml']

app = Flask(__name__)
services = {

    'kmeans': kmeans,
    'gauss': gauss,

}

cors = CORS(app, resources={
    r'/{}'.format(service): {"origins": "*"} for service in services
})

@app.route('/', methods=['GET'])
def test():
    return "Hello"

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data = request.files['data']
        ext = data.filename.split('.')[1]
        if(ext in exts):
            data.save('uploads/' + data.filename)
            return 'File saved to uploads directory@'
        else:
            return 'File type not accepted!'
    return render_template('upload.html')


@app.route('/<string:service_name>', methods=['GET','POST'])
def service(service_name):
    try:
        service_func = services[service_name]
    except:
        # service does not exist
        return None, 401
    
    data = request.get_json()
    output_data = service_func(data)
    return jsonify(output_data)
    #return render_template('index.html',output=output_data)

# @app.route('/predict', methods=['GET','POST'])
# def predict():
	
# 	# Receives the input query from form
# 	if request.method == 'POST':
# 		namequery = request.form['namequery']
#         # model=request.form['modelname']
#         # modelload=open("models/"+str(model)+".pkl","rb")
#         # clf = joblib.load(modelload)
# 		data = [namequery]
# 		vect = cv.transform(data).toarray()
# 		my_prediction = clf.predict(vect)
# 	return render_template('results.html',prediction = my_prediction,name = namequery.upper())

if __name__ == "__main__":
    app.run(debug=True)
