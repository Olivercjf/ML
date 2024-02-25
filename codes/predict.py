from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

model = YOLO('yolov8n-seg-custom.pt')  # Load model

model.predict(source='../data/prueba_cm/*.png', show=True, save=True, hide_labels=False, box=False,
               hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2) # Parameters

ConfusionMatrix(nc=1)  # Create a confusion matrix