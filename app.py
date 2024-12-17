from flask import Flask, render_template, jsonify
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2


app = Flask(__name__)

model = YOLO("C://Workspace//FYP//RT-SignLanguage-YOLOv8//Basic-1st.pt")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def start_detection():
    try:
        model.predict(source="0", show=True, conf=0.6)
        return render_template('completed.html')
    except Exception as e:
        return render_template('error.html', error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True)