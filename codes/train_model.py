from ultralytics import YOLO

# Load the model and define its configuration
model = YOLO('./best.pt')

# Train the model
results = model.val(data='../data/butterflies_example.yaml', task='segment', epochs=10, imgsz=320, batch=8, save_json=False)

print(results.confusion_matrix.matrix)