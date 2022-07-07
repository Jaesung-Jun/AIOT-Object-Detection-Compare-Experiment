import numpy as np
import tflite_runtime.interpreter as tflite
import osAdvanced
from PIL import Image
from PIL import ImageDraw
import json

with open('imagenet-simple-labels.json') as f:
    labels = json.load(f)

DEFAULT_DIR = "./"
DATASET_DIR = "../dataset/coco-2017-val/val2017/"
NUM_OF_IMAGES = 500

image_paths = osAdvanced.File_Control.searchAllFilesInDirectory(DATASET_DIR, "jpg", progressbar=True)

images = np.ndarray((NUM_OF_IMAGES, 320, 320, 3), dtype=np.uint8)

for i in range(NUM_OF_IMAGES):
    image = Image.open(image_paths[i])
    if len(np.array(image).shape) == 3:
        print(image_paths[i])
        images[i] = image.resize((320, 320))
        images[i] = np.array(images[i])
        images[i] = np.reshape(np.array(images[i], dtype=np.uint8), (1, 320, 320, 3))

edge_tpu_delegate = None
try:
    edge_tpu_delegate = tflite.load_delegate('libedgetpu.so.1.0')
    print("EdgeTPU Detected.")
except:
    print("No EdgeTPU detected. Falling back to CPU.")

if edge_tpu_delegate is None:
    interpreter = tflite.Interpreter(model_path=f"{DEFAULT_DIR}lite-model_efficientdet_lite0_int8_1.tflite")
else:
    interpreter = tflite.Interpreter(model_path=f"{DEFAULT_DIR}lite-model_efficientdet_lite0_int8_1.tflite", experimental_delegates=[edge_tpu_delegate])

#interpreter = tflite.Interpreter(model_path=f"{DEFAULT_DIR}lite-model_efficientdet_lite0_int8_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

for i in range(NUM_OF_IMAGES):
    input_data = np.expand_dims(images[i], axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = {
                    'detection_boxes': '',
                    'detection_classes': '',
                    'detection_scores': '',
                    'num_detections': '',
                }

    output_data['detection_boxes'] = interpreter.get_tensor(output_details[0]['index'])[0]
    output_data['detection_classes'] = interpreter.get_tensor(output_details[1]['index'])[0]
    output_data['detection_scores'] = interpreter.get_tensor(output_details[2]['index'])[0]
    output_data['num_detections'] = interpreter.get_tensor(output_details[3]['index'])

    THRESHOLD = 0
    score_indices = np.where(output_data['detection_scores'] > THRESHOLD)
    print("Image Path : ", image_paths[i])
    for score_index in score_indices[0]:
        #print("Image Path : ", image_paths[i])
        #print("Detected Box(x,y,w,h) : ", output_data['detection_boxes'][score_index]*320)
        #print("Detected Class : ", output_data['detection_classes'][score_index])
        #print("Confidence Score : ", output_data['detection_scores'][score_index])
