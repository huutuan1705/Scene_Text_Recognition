import os
import xml.etree.ElementTree as ET 

def extract_data_from_xml(root_dir):
    xml_path = os.path.join(root_dir, "words.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []
    
    