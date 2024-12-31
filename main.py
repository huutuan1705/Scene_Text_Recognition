from sklearn.model_selection import train_test_split
from extract_text_block.train_yolov11 import training
from extract_text_block.utils import preprocess_data

def training_yolo():
    seed = 0
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True
    
    yolo_data, _ = preprocess_data("SceneTrialTrain")
    training_data, test_data = train_test_split(
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