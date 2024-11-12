# import os
# import torch
# import numpy as np
# from torchvision import models, transforms
# from torchvision.models import MobileNet_V2_Weights
# from PIL import Image
# import ray
# import time
# import pandas as pd

# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.applications.mobilenet_v2 import (
# #     preprocess_input,
# #     decode_predictions,
# #     MobileNetV2,
# # )

# # Ray initialization
# ray.init(address='auto')
# print("Ray initialized")


# # Preprocessing function for images
# preprocess = transforms.Compose([
#     transforms.Resize(160),
#     transforms.CenterCrop(160),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def load_model():
#     # model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
#     model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

#     model.eval()  # Set the model to evaluation mode
#     # scripted_model = torch.jit.script(model)
#     return model

# def preprocess_image(image_path): ## this is for tensorflow
#     img = image.load_img(image_path, target_size=(224, 224))  # Load and resize image
#     img_array = image.img_to_array(img)  # Convert to numpy array
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return preprocess_input(img_array)  # Preprocess input for MobileNetV2

# # Image inference function
# def infer_image(image_path,model):
#     img = Image.open(image_path)
#     if img.mode != "RGB":
#         img = img.convert("RGB")
#     img = preprocess(img)  # Apply preprocessing
#     img = img.unsqueeze(0)  # Add batch dimension
#     start_time = time.time()
#     with torch.no_grad():  # Inference without tracking gradients
#         preds = model(img)
#     end_time = time.time()
#     # Decode predictions (get top-1 prediction)
#     _, predicted_class = torch.max(preds, 1)
    
#     return {"detections": predicted_class.item(), "inference time": end_time - start_time}

# @ray.remote
# def run_inference_on_directory(image_dir):
#     model=load_model()
#     # model = ray.get(model_ref)  # Retrieve the model from the object store
#     results = {}
#     response_times = []
#     response_times_path = os.getenv("RESPONSE_TIME_PATH", "response_times.csv")

# # Check if the directory exists, and create it if not
#     i=0
#     for img_file in os.listdir(image_dir):
#         img_path = os.path.join(image_dir, img_file)
#         if os.path.isfile(img_path):
#             start_time = time.time()
#             predicted_class = infer_image(img_path,model) ## this is for pytorch
#             # predicted_class, predictions = infer_image_tf(img_path, model) ## this is for tensorflow
#             end_time = time.time()

#             results[img_file] = {"class": predicted_class, "time": end_time - start_time}
#             response_times.append(end_time - start_time)
#         i+=1
#         # if i==100:
#         #     break

#     # Save response times to a CSV file
#     try:
#         pd.DataFrame(response_times, columns=["response_time"]).to_csv(response_times_path, index=False)
#         print("Response times saved successfully.")
#     except Exception as e:
#         print(f"Error saving response times: {e}")

#     return results

# if __name__ == "__main__":
#     image_dir = "mobilenet-imagenet/images/test/"  # Change to your directory containing images
#     print("Running inference on images in directory:", image_dir)
    
#     inference_results = ray.get(run_inference_on_directory.remote(image_dir))
    
#     for image_file, prediction in inference_results.items():
#         print(f"Image: {image_file}, Prediction: {prediction['class']}")
#     # print("inference results",inference_results)

# import os
# import torch
# import numpy as np
# from torchvision import models, transforms
# from torchvision.models import MobileNet_V2_Weights
# from PIL import Image
# import ray
# import time
# import pandas as pd

# # Apply real-time scheduling to the current main process
# print(f"Applying real-time scheduling to the main process (PID: {os.getpid()})")
# result = os.system(f'chrt -r 99 -p {os.getpid()}')
# if result != 0:
#     print("Failed to set real-time scheduling policy for the main process. Check if `chrt` is available and permissions are sufficient.")
# else:
#     print("Successfully applied real-time scheduling to the main process.")
# os.system(f'chrt -p {os.getpid()}')  # Verify the change

# # Ray initialization
# ray.init(address='auto')
# print("Ray initialized")

# # Preprocessing function for images
# preprocess = transforms.Compose([
#     transforms.Resize(160),
#     transforms.CenterCrop(160),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def load_model():
#     model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
#     model.eval()  # Set the model to evaluation mode
#     return model

# def preprocess_image(image_path):
#     img = Image.open(image_path)
#     if img.mode != "RGB":
#         img = img.convert("RGB")
#     img = preprocess(img)  # Apply preprocessing
#     img = img.unsqueeze(0)  # Add batch dimension
#     return img

# def infer_image(image_path, model):
#     img = preprocess_image(image_path)
#     start_time = time.time()
#     with torch.no_grad():  # Inference without tracking gradients
#         preds = model(img)
#     end_time = time.time()
#     _, predicted_class = torch.max(preds, 1)  # Get top-1 prediction
#     return {"detections": predicted_class.item(), "inference time": end_time - start_time}

# @ray.remote
# def run_inference_on_directory(image_dir):
#     # Apply real-time scheduling to the Ray worker process
#     print(f"Applying real-time scheduling to Ray worker (PID: {os.getpid()})")
#     result = os.system(f'chrt -r 99 -p {os.getpid()}')
#     if result != 0:
#         print("Failed to set real-time scheduling policy for Ray worker. Check if `chrt` is available and permissions are sufficient.")
#     else:
#         print("Successfully applied real-time scheduling to Ray worker.")
#     os.system(f'chrt -p {os.getpid()}')  # Verify the change

#     model = load_model()
#     results = {}
#     response_times = []
#     response_times_path = os.getenv("RESPONSE_TIME_PATH", "response_times.csv")

#     for img_file in os.listdir(image_dir):
#         img_path = os.path.join(image_dir, img_file)
#         if os.path.isfile(img_path):
#             start_time = time.time()
#             predicted_class = infer_image(img_path, model)
#             end_time = time.time()

#             results[img_file] = {"class": predicted_class, "time": end_time - start_time}
#             response_times.append(end_time - start_time)

#     # Save response times to a CSV file
#     try:
#         pd.DataFrame(response_times, columns=["response_time"]).to_csv(response_times_path, index=False)
#         print("Response times saved successfully.")
#     except Exception as e:
#         print(f"Error saving response times: {e}")

#     return results

# if __name__ == "__main__":
#     image_dir = "mobilenet-imagenet/images/test/"  # Directory containing images
#     print("Running inference on images in directory:", image_dir)

#     inference_results = ray.get(run_inference_on_directory.remote(image_dir))

#     for image_file, prediction in inference_results.items():
#         print(f"Image: {image_file}, Prediction: {prediction['class']}")

import os
import ray
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import time
import pandas as pd
import subprocess

import ctypes
from ctypes.util import find_library

# def apply_and_check_scheduling():
#     """Applies and verifies real-time scheduling using `chrt`."""
#     try:
#         pid = os.getpid()
#         print(f"Applying `chrt` real-time scheduling to PID {pid}...")

#         os.system(f"ps -p {pid}")
#         os.system(f"chrt -r -p 99 {pid}")

#         check_result = subprocess.run(["chrt", "-p", str(pid)], capture_output=True, text=True)
#         print(f"Scheduling policy verification for PID {pid}:")
#         print(check_result.stdout)
#     except Exception as e:
#         print(f"Error applying `chrt`: {e}")

def set_sched_rr_all_threads(priority=90):
    libc = ctypes.CDLL(find_library("c"), use_errno=True)
    SCHED_RR = 2

    class SchedParam(ctypes.Structure):
        _fields_ = [("sched_priority", ctypes.c_int)]

    main_pid = os.getpid()
    task_dir = f"/proc/{main_pid}/task/"
    
    try:
        for tid in os.listdir(task_dir):
            tid = int(tid)
            param = SchedParam(sched_priority=priority)
            result = libc.sched_setscheduler(tid, SCHED_RR, ctypes.pointer(param))
            if result != 0:
                err = ctypes.get_errno()
                print(f"Failed to set SCHED_RR for TID {tid}. Error code: {err} - {os.strerror(err)}")
            else:
                print(f"Successfully set SCHED_RR for TID {tid}.")
    except Exception as e:
        print(f"Error applying `SCHED_RR` to all threads: {e}")


print("Initializing Ray...")
try:
    ray.init(address='auto')
    print("Ray initialized successfully.")
except Exception as e:
    print(f"Ray initialization failed: {e}")

# Preprocessing function for images
preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    print("Loading model...")
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    print("Model loaded.")
    return model

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = preprocess(img)
    return img.unsqueeze(0)

def infer_image(image_path, model):
    img = preprocess_image(image_path)
    start_time = time.time()
    with torch.no_grad():
        preds = model(img)
    end_time = time.time()
    _, predicted_class = torch.max(preds, 1)
    return {"detections": predicted_class.item(), "inference time": end_time - start_time}

@ray.remote
def run_inference_on_directory(image_dir):
    print(f"Ray worker process started with PID: {os.getpid()}")
    
    # Apply and check scheduling for the Ray worker process
    # set_sched_rr_all_threads(priority=99)

    model = load_model()
    results = {}
    response_times = []
    response_times_path = os.getenv("RESPONSE_TIME_PATH", "response_times.csv")

    print("Running inference on images in directory")
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            start_time = time.time()
            predicted_class = infer_image(img_path, model)
            end_time = time.time()
            results[img_file] = {"class": predicted_class, "time": end_time - start_time}
            response_times.append(end_time - start_time)
    print("Inference completed.")

    try:
        pd.DataFrame(response_times, columns=["response_time"]).to_csv(response_times_path, index=False)
        print("Response times saved successfully.")
    except Exception as e:
        print(f"Error saving response times: {e}")

    return results

if __name__ == "__main__":
    
    image_dir = "mobilenet-imagenet/images/test/"
    print(f"Running inference on images in directory: {image_dir}")

    try:
        inference_results = ray.get(run_inference_on_directory.remote(image_dir))
        for image_file, prediction in inference_results.items():
            print(f"Image: {image_file}, Prediction: {prediction['class']}")
    except Exception as e:
        print(f"Inference error: {e}")
