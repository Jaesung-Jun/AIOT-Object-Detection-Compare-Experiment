import numpy as np
import tflite_runtime.interpreter as tflite
import osAdvanced
from PIL import Image
import json
import os

import argparse

parser = argparse.ArgumentParser(description="EfficientDet Test Code (Edited 2022-05-06)")
parser.add_argument('--start_index', type=int, help='start_index')
parser.add_argument('--end_index', type=int, help='end_index')
parser.add_argument('--threshold', type=float, help='threshold value', default=0.5)

args = parser.parse_args()

DEFAULT_DIR = "./"
DATASET_DIR = "../dataset/coco-2017-val/val2017/"
SAVE_DIR = "./result"
START_INDEX = int(args.start_index)
END_INDEX = int(args.end_index)    #5000
NUM_OF_IMAGES = END_INDEX-START_INDEX

image_paths = osAdvanced.File_Control.searchAllFilesInDirectory(DATASET_DIR, "jpg", progressbar=True)
image_paths.sort()
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

import time
sum_time = 0
for i in range(NUM_OF_IMAGES):
    #print("Image Path : ", image_paths[i+START_INDEX])    
    input_data = np.expand_dims(images[i], axis=0)
    
    s_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_array = interpreter.get_tensor(output_details[0]['index'])
    sum_time+=time.time()-s_time

fps = sum_time / float(NUM_OF_IMAGES)
now_time = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time()))

f = open(f"./yolov5_inference_time.txt", "a")
f.write(f"{now_time} [start_index : {args.start_index}, end_index :{args.end_index}]\n")
f.write(f"Inference time : {sum_time}\n")
f.write(f"Frame Per Second(fps) : {fps}\n")

f.close()
