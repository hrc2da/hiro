from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import atexit


app = Flask(__name__)
CORS(app)

counter = 0
cap = None
@app.route("/")
def hello():
    return "This is Sparta!"

@app.route("/capture", methods=['POST'])
def capture():
    filename = request.form['filename']
    global counter
    global cap
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'images/{filename}.png',gray)
        counter += 1
        return {'success': True}, 200
    else:
        return {'success': False}, 500

def shutdown():
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = 0
    cap = cv2.VideoCapture(0)
    atexit.register(shutdown)
    app.run(host='0.0.0.0',port=5001)
    