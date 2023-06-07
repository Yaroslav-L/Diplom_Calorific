import tkinter
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
import time
import numpy as np
import warnings
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import six
import json

matplotlib.use('TkAgg')

def object_detect(img):

    IMAGE_SIZE = (12, 8) 
    PATH_TO_SAVED_MODEL="Diplom_Calorific\Server\model\saved_model"

    img = img.convert('RGB')

    print('Loading model...', end='')
    detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print('Done!')
    category_index=label_map_util.create_category_index_from_labelmap("Diplom_Calorific\Server\model\label_map.pbtxt",use_display_name=True)

    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.4, # Adjust this value to set the minimum probability boxes to be classified as True
        agnostic_mode=False)
    

#    plt.figure(figsize=IMAGE_SIZE, dpi=200)
#    plt.axis("off")
#    plt.imshow(image_np_with_detections)
#    plt.show()

    return(detections)


def output_data(detections):
    all_name = ""
    kcal = 0
    boxes = detections['detection_boxes']
    det = detections['detection_classes']
    max_boxes_to_draw = boxes.shape[0]
    scores = detections['detection_scores']
    min_score_thresh=.7
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            with open("Diplom_Calorific\Server\model\\fkcal_data_objdtc.json",'r',encoding='utf-8') as f:
                d = json.loads(f.read())
                x = d[str(det[i])]
                all_name = all_name+x["name"]+", "
                kcal = kcal + int(x["KcaL"])
    kcal = str(kcal)
    all_p = json.dumps({"name":all_name,"KcaL":kcal}, sort_keys=True, ensure_ascii=False)
    
    return(all_p)

                
def main_object_detect(img):
    return(json.loads(output_data(object_detect(img))))


#if __name__ == "__main__":     
#    print(main_object_detect(Image.open('image2.jpg')))

