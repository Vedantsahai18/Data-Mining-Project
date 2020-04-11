from modules.lin_regress import linRegression
from modules.svm import svm
from modules.kmeans import kmeans
from modules.kmedoids import kmedoids
from modules.lda import lda
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os.path

save_path = '/uploads/'
exts = ['csv', 'json', 'yaml']

app = Flask(__name__)
services = {
    'svm': svm,
    'lin_regress': linRegression,
    'kmeans': kmeans,
    'kmedoids': kmedoids,
    'lda': lda
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


@app.route('/<string:service_name>', methods=['POST'])
def service(service_name):
    try:
        service_func = services[service_name]
    except:
        # service does not exist
        return None, 401
    
    data = request.get_json()
    output_data = service_func(data)
    return jsonify(output_data)


if __name__ == "__main__":
    app.run(debug=True)
