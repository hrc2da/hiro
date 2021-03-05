import flask
from flask import request, jsonify, abort
from nlp_utils import NoteParser


from flask import Flask
app = Flask(__name__)

@app.route('/')
def main_page():
    return app.send_static_file('index.html')

@app.route('/txt2embedding', methods=['GET', 'POST'])
def get_text2embedding():
    if request.method == 'POST':
        txt = request.form['word']
        return str(parser.txt2embedding(txt))

@app.route('/txt2pca_km', methods=['GET', 'POST'])
def get_text2pca():
    print("got the request")
    if request.method == 'POST':
        # txt = request.json['word']
        allwords = request.json['allwords']
        try:
            coords = parser.txt2localpca(allwords).tolist()
            clusters = []
            if len(allwords) > 2:
                _, clusters = parser.txt2clusters(allwords)
            return jsonify({"pca":coords, "kmeans":clusters})
        except KeyError as e:
            abort(404,str(e))

@app.route('/photo2pca', methods = ['POST'])
def get_photo2word():    
    if request.method == 'POST':
        f = request.files['image']
        f.save('data/tmp/latest.jpg')
        word = parser.photo2txt('data/tmp/latest.jpg')
        print(word)
        return word


# @app.route('/photo2embedding', methods = ['POST'])
# def get_photo2embedding():    
#     if request.method == 'POST':
#         f = request.files['the_file']
#         f.save('data/tmp/latest.jpg')

if __name__=='__main__':
    parser = NoteParser()
    app.run()