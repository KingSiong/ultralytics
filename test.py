from ultralytics import YOLO

# Load a model
model = YOLO('xjtun.yaml').load('yolov8n.pt')  # xjtu model
# model = YOLO('/home/sjs/lightweight-object-detection/ultralytics/runs/detect/train8/weights/best.pt')
# model = YOLO('yolov6n.yaml') # yolov6n # cannot be used
# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8n.yaml')

# train the model
data = 'coco.yaml'
epochs = 100
imgsz = 640
devices = '2, 3'
batch_size = 64
freeze_list = [0, 1, 2, 3, 4, 5, 6, 7]
# freeze_list = [_ for _ in range(23)]
# del freeze_list[8]

results = model.train(data=data, epochs=epochs, batch=batch_size, imgsz=imgsz, freeze=freeze_list, device=devices)

# Validate the model
# metrics = model.val(batch=20, device=0, data='coco.yaml')  # no arguments needed, dataset and settings remembered

# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category