import os

from flask import Flask, request, flash, redirect, render_template, url_for
from werkzeug.utils import secure_filename

from kmeans import elbow, parse, kmeans

UPLOAD_FOLDER = 'csv'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/algo', methods = ['GET', 'POST'])
def algo():
    if request.method == 'POST':
        filename = request.form['filename']
        if request.form['algorithm'] == "DB-SCAN":
            chosen_algo=request.form['algorithm']
            
            return render_template('result.html', image1 = 'elbow.png')
        elif request.form['algorithm'] == "kmean":
            X = parse(filename, 3, 4)
            elbow(X)
            kmeans(X, n_clusters=5)
            text = open('static\summary.txt', 'r+')
            content = text.read()
            text.close()
            return render_template('result.html', image1 = 'elbow.png', image2 = 'kmeans_clusters.png',text=content)
        return 'Nothing Selected'
        


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('algo.html', filename = filename)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run()
