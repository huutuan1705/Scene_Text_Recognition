import os
import yaml

from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from extract_text_block.train_yolov11 import training
from extract_text_block.utils_yolo import preprocess_data, save_data

def training_yolo(dataset_dir="SceneTrialTrain"):
    seed = 0
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True
    
    yolo_data, data_yaml = preprocess_data(dataset_dir)
    train_data, test_data = train_test_split(
        yolo_data,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )
    
    test_data, val_data = train_test_split(
        test_data,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle
    )
    
    save_yolo_data_dir = "datasets/yolo_data"
    os.makedirs(save_yolo_data_dir, exist_ok=True)
    save_train_dir = os.path.join(save_yolo_data_dir, "train")
    save_val_dir = os.path.join(save_yolo_data_dir, "val")
    save_test_dir = os.path.join(save_yolo_data_dir, "test")
    
    save_data(train_data, dataset_dir, save_train_dir)
    save_data(test_data, dataset_dir, save_test_dir)
    save_data(val_data, dataset_dir, save_val_dir)
    
    yolo_yaml_path = os.path.join(save_yolo_data_dir, "data.yaml")
    with open(yolo_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    model = YOLO("yolov11m.pt")
    results = model.train(
        data=data_yaml, epochs=100, imgsz=640, cache=True, patience=20, plots=True
    )
    
    