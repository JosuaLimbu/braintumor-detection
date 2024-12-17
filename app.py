from flask import Flask, request, render_template
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
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

with open("model/name.txt", "r") as f:
    single_class_name = f.readline().strip()

register_coco_instances(
    "my_dataset_test", {}, 
    "path/to/test/_annotations.coco.json", 
    "path/to/test"
)
test_metadata = MetadataCatalog.get("my_dataset_test")
MetadataCatalog.get("my_dataset_test").thing_classes = [single_class_name]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./model/model.pth"  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.DEVICE = "cpu" 
predictor = DefaultPredictor(cfg)

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    input_file = None
    output_file = None

    url = "https://assetsbraintumor.vercel.app/content.html"
    
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
    else:
        html_content = "<p>Error fetching content from URL 1</p>"
    
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                input_file = os.path.join('uploads', filename).replace("\\", "/")  

                if 'predict' in request.form:
                    img = cv2.imread(upload_path)
                    outputs = predictor(img)

                    instances = outputs["instances"].to("cpu")
                    instances.pred_classes = np.zeros_like(instances.pred_classes.numpy())  

                    v = Visualizer(
                        img[:, :, ::-1], 
                        metadata=test_metadata, 
                        scale=0.8
                    )
                    out = v.draw_instance_predictions(instances)

                    result_filename = 'result_' + filename
                    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                    cv2.imwrite(result_path, out.get_image()[:, :, ::-1])
                    output_file = os.path.join('results', result_filename).replace("\\", "/")  

    return render_template('index.html', html_content=html_content, input_file=input_file, output_file=output_file)

if __name__ == '__main__':
    app.run(debug=True)