import os

from flask import Flask, request, flash, redirect, render_template, url_for
from werkzeug.utils import secure_filename

from modules.birch import train as bt
from modules.dbscan import train as dbt
from modules.optics import train as opt
from modules.spectral_clustering import train as sct
from modules.affinity_propagation import train as apt

UPLOAD_FOLDER = 'uploads'
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
            print(('uploads/'+str( filename)))
            print(dbt(('uploads/'+str( filename)),0.30) )    
            return render_template('result.html', image1 = 'dbscan.png',image2='dbscan_unlabelled_data.png')
        
        # elif request.form['algorithm'] == "birch":
        #     bt(('uploads/'+str(filename)),2)  
        #     return render_template('result.html', image1 = 'birch.png', image2 = 'birch_unlabelled.png')
        
        # elif request.form['algorithm'] == "optics":
        #     opt(('uploads/'+str(filename)),0.80)  
        #     return render_template('result.html', image1 = 'optics.png', image2 = 'optics_unlabelled.png')
        
        # elif request.form['algorithm'] == "affprop":
        #     apt(('uploads/'+str(filename)))
        #     return render_template('result.html', image1 = 'affinity_propagation.png', image2 = 'affinity_propagation_unlabelled.png')
        
        # elif request.form['algorithm'] == "specclust":
        #     sct(('uploads/'+str(filename)),2)  
        #     return render_template('result.html', image1 = 'spectral_clustering.png', image2 = 'spectral_clustering_unlabelled.png')    
        
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
    app.run(debug=True)
