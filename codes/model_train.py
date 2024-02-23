from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(task='segment', epochs=50, data='../data/dataset2.yaml', imgsz=640, batch=8)