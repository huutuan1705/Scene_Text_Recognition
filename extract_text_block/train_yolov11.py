from ultralytics import YOLO

def training(yolo_yaml_path, epochs=100, imgsz=640):
    model = YOLO("yolo11m.pt")
    
    results = model.train(
        data=yolo_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        patience=20,
        plots=True
    )
    
    return results

def evaluate(model_path):
    model = YOLO(model_path)
    metrics = model.val()
    
    return metrics