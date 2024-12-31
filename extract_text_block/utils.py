import os
import shutil
import xml.etree.ElementTree as ET 

def extract_data_from_xml(root_dir):
    xml_path = os.path.join(root_dir, "words.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []
    
    for img in root:
        bbs_of_img = []
        label_of_img = []
        
        for bbs in img.findall('taggedRectangles'):
            for bb in bbs:
                # chech non-alphabet and non-number
                if not bb[0].text.isalnum():
                    continue
                if ' ' in bb[0].text.lower() or '   ' in bb[0].text.lower():
                    continue
                
                bbs_of_img.append(
                    [
                        float(bb.attrib('x')),
                        float(bb.attrib('x')),
                        float(bb.attrib('width')),
                        float(bb.attrib('height')),
                    ]
                )
                label_of_img.append(bb[0].text.lower())
        
        img_path = os.path.join(root_dir, img[0].text)
        img_paths.append(img_path)
        img_sizes.append(int(img[1].attrib('x'), int(img[1].attrib('y'))))
        bboxes.append(bbs_of_img)
        img_labels.append(label_of_img)
    
    return img_paths, img_sizes, img_labels, bboxes

def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []
    
    for image_path, image_size, bboxes in zip(image_paths, image_sizes, bounding_boxes):
        image_width, image_height = image_size
        
        yolo_labels = []
        for bbox in bboxes:
            x, y, w, h = bbox
            
            # Calculate normalized bounding box coordinates
            center_x = (x+w/2) / image_width
            center_y = (y+h/2) / image_height
            
            normalized_width = w / image_width
            normalized_height = h / image_height
            
            # We have only 1 class, we set class_id to 0
            class_id = 0
            
            yolo_label = f"{class_id} {center_x} {center_y} {normalized_width} {normalized_height}"
            yolo_labels.append(yolo_label)
        
        yolo_data.append((image_path, yolo_labels))
        
    return yolo_data

def save_data(data, src_img_dir, save_dir):
    # create folder if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Make images and labels folder
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)
    
    for image_path, yolo_labels in data:
        # Copy image to images folder
        shutil.copy(os.path.join(src_img_dir, image_path), os.path.join(save_dir, "images"))
        
        # Save labels to labels folder
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]
        
        with open(os.path.join(save_dir, "labels", f"{image_name}.txt"), "w") as f:
            for label in yolo_labels:
                f.write(f"{label}\n")
                
def preprocess_data(root_dir):
    img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(root_dir)
    
    class_labels = ["text"]
    yolo_data = convert_to_yolo_format(img_paths, img_sizes, bboxes)
    
    data_yaml = {
        "path": ".dataset/yolo_data",
        "train": "train/images",
        "test": "test/images",
        "val": "val/images",
        "nc": 1,
        "names": class_labels
    }
    
    return yolo_data, data_yaml
    