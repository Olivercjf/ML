from ultralytics import YOLO

model = YOLO('/home/oliver18/Documents/ML/ML/codes/best.pt')  # Load model

model.predict(source='../data/test_folder/*.jpg', show=True, save=True, hide_labels=False, box=False,
               hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2) # Parameters