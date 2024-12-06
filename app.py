from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register dataset
register_coco_instances(
    "my_dataset_test", {}, 
    "path/to/test/_annotations.coco.json",  # Ganti dengan path lokal
    "path/to/test"  # Ganti dengan path folder dataset 
)
test_metadata = MetadataCatalog.get("my_dataset_test")

# Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./model/model.pth"  # Path ke model hasil training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold untuk deteksi
cfg.MODEL.DEVICE = "cpu"  
predictor = DefaultPredictor(cfg)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load image and make predictions
    img = cv2.imread(file_path)
    outputs = predictor(img)

    # Visualize predictions
    v = Visualizer(
        img[:, :, ::-1], 
        metadata=test_metadata, 
        scale=0.8
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the output image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    # Display the image
    return render_template('result.html', output_file='output_' + filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
