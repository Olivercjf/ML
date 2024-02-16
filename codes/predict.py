from ultralytics import YOLO

model = YOLO('yolov8n-seg-custom.pt')  # Load model

model.predict(source='../data/2.mp4', show=True, save=True, hide_labels=False, box=False,
               hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2) # Parameters