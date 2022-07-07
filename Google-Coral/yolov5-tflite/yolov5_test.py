import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import json

with open('imagenet-simple-labels.json') as f:
    labels = json.load(f)

DEFAULT_DIR = "./"

#Load Image
cute_pug = Image.open("./test_images/pug_dog.jpg")
cute_pug = cute_pug.resize((320, 320))
cute_pug = np.reshape(np.array(cute_pug, dtype=np.float32), (1, 320, 320, 3))
print(cute_pug.shape)

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=f"{DEFAULT_DIR}lite-model_yolo-v5-tflite_tflite_model_1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)

input_data = cute_pug

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

for t in output_data[0][0]:
    print(t)

print("index : " + str(np.argmax(output_data[0][0])-1))
print("label : " + labels[np.argmax(output_data[0][0])-1])
