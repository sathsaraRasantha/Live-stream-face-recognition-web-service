from flask import Flask, render_template, Response
import cv2


#EXTRA PART
from PIL import Image
from keras.models import load_model
import numpy as np
model = load_model('facefeatures_new_model.h5')
#END OF EXTRA PART


#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray,1.3,5)
            
            for (x,y,w,h) in face_rects:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

