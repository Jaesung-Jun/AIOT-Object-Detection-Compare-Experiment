import numpy as np
import tflite_runtime.interpreter as tflite
import osAdvanced
from PIL import Image
from PIL import ImageDraw
import json
import os

import argparse

def class_filter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def yolo_detect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = class_filter(output_data[..., 5:]) # get classes

    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]
    #xywh = [x, y, w, h]
    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

parser = argparse.ArgumentParser(description="EfficientDet Test Code (Edited 2022-05-06)")
parser.add_argument('--start_index', type=int, help='start_index')
parser.add_argument('--end_index', type=int, help='end_index')
parser.add_argument('--threshold', type=float, help='threshold value', default=0.5)

args = parser.parse_args()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error Creating Ditectory : " + ditectory)

with open('imagenet-simple-labels.json') as f:
    labels = json.load(f)

DEFAULT_DIR = "./"
DATASET_DIR = "../dataset/coco-2017-val/val2017/"
SAVE_DIR = "./result/"
START_INDEX = int(args.start_index)
END_INDEX = int(args.end_index)    #5000
NUM_OF_IMAGES = END_INDEX-START_INDEX

image_paths = osAdvanced.File_Control.searchAllFilesInDirectory(DATASET_DIR, "jpg", progressbar=True)
image_paths.sort()
print(len(image_paths))
images = np.ndarray((NUM_OF_IMAGES, 320, 320, 3), dtype=np.float32)

for i in range(START_INDEX, END_INDEX):
    image = Image.open(image_paths[i])
    if len(np.array(image).shape) == 3:
        print(str(i) + " : " + image_paths[i])
        images[i-START_INDEX] = image.resize((320, 320))
        images[i-START_INDEX] = np.array(images[i-START_INDEX])
        images[i-START_INDEX] = np.reshape(np.array(images[i-START_INDEX], dtype=np.uint8), (1, 320, 320, 3))

edge_tpu_delegate = None

try:
    edge_tpu_delegate = tflite.load_delegate('libedgetpu.so.1.0')
    print("EdgeTPU Detected.")
except:
    print("No EdgeTPU detected. Falling back to CPU.")

if edge_tpu_delegate is None:
    interpreter = tflite.Interpreter(model_path=f"{DEFAULT_DIR}lite-model_yolo-v5-tflite_tflite_model_1.tflite")
else:
    interpreter = tflite.Interpreter(model_path=f"{DEFAULT_DIR}lite-model_yolo-v5-tflite_tflite_model_1.tflite", experimental_delegates=[edge_tpu_delegate])

#interpreter = tflite.Interpreter(model_path=f"{DEFAULT_DIR}lite-model_efficientdet_lite0_int8_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

ANNOTATION_DIR = f'./predicted_annotation_{START_INDEX}_{END_INDEX}'

createFolder(ANNOTATION_DIR)

for i in range(NUM_OF_IMAGES):
    input_data = np.expand_dims(images[i], axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = {
                    'detection_boxes': '',
                    'detection_classes': '',
                    'detection_scores': '',
                    #'num_detections': '',
                }
    output_array = interpreter.get_tensor(output_details[0]['index'])
    xyxy, classes, scores = yolo_detect(output_array)
    
    output_data['detection_boxes'] = xyxy
    output_data['detection_classes'] = classes
    output_data['detection_scores'] = scores
    #output_data['num_detections'] = interpreter.get_tensor(output_details[3]['index'])

    THRESHOLD = args.threshold
    score_indices = np.where(output_data['detection_scores'] > THRESHOLD)  

    print("Image Path : ", image_paths[i+START_INDEX])

    #f = open(f"{ANNOTATION_DIR}/{os.path.splitext(os.path.basename(image_paths[i+START_INDEX]))[0]}.txt", "w")
    res = []
    
    for score_index in score_indices[0]:
        #print("Image Path : ", image_paths[i])
        #print("Detected Box(x,y,w,h) : ", output_data['detection_boxes'][score_index]*320, "....")

        #print("Detected Class : ", output_data['detection_classes'][score_index])
        #print("Confidence Score : ", output_data['detection_scores'][score_index])
        bbox_coord = [output_data['detection_boxes'][0][score_index]*320, 
                      output_data['detection_boxes'][1][score_index]*320, 
                      output_data['detection_boxes'][2][score_index]*320, 
                      output_data['detection_boxes'][3][score_index]*320]
        
        res.append([bbox_coord, output_data['detection_classes'][score_index], output_data['detection_scores'][score_index]])

        #bbox_coords = output_data['detection_boxes'][score_index]*320
        # (x, y, w, h) =====> (x_center, y_center, width, height)
        #bbox_coords_new = [(bbox_coords[0]+(bbox_coords[2]/2)), (bbox_coords[1]+(bbox_coords[3]/2)), bbox_coords[2], bbox_coords[3]]
        #f.write(f"{int(output_data['detection_classes'][score_index])} {output_data['detection_scores'][score_index]} {bbox_coords_new[0]} {bbox_coords_new[1]} {bbox_coords_new[2]} {bbox_coords_new[3]}\n")
    #f.close()

print("length :", len(res))

import time
import joblib

now_time = time.strftime('%m_%d_%H-%M', time.localtime(time.time()))
joblib.dump(res, f"{SAVE_DIR}yolov5_{now_time}_{args.start_index}-{args.end_index}.pkl")
print(f"Saved : {SAVE_DIR}yolov5_{now_time}.pkl")


