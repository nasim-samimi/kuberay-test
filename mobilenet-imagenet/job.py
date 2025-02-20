import os
import ray
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights,efficientnet_b0, EfficientNet_B0_Weights
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
    # model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.eval()
    print("Model loaded.")
    return model

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = preprocess(img)
    return img.unsqueeze(0)

def infer_image(img, model):
    start_time = time.time()
    with torch.no_grad():
        preds = model(img)
    end_time = time.time()
    _, predicted_class = torch.max(preds, 1)
    return {"detections": predicted_class.item(), "inference time": end_time - start_time}

@ray.remote
def run_inference_on_directory(image_dir):
    print(f"Ray worker process started with PID: {os.getpid()}")

    model = load_model()
    results = {}
    response_times = []
    response_times_path = os.getenv("RESPONSE_TIME_PATH", "response_times.csv")

    print("Running inference on images in directory")
    i=0
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            img = preprocess_image(img_path)
            start_time = time.time()
            predicted_class = infer_image(img, model)
            end_time = time.time()
            results[img_file] = {"class": predicted_class, "time": end_time - start_time}
            response_times.append(end_time - start_time)
        i+=1
        if (i%100)==0:
            print(i)
    print("Inference completed.")

    try:
        pd.DataFrame(response_times, columns=["response_time"]).to_csv(response_times_path, index=False)
        print("Response times saved successfully.")
    except Exception as e:
        print(f"Error saving response times: {e}")

    return results

@ray.remote(num_cpus=1)  # ⚡ Explicit resource allocation for Ray workers
def run_batch_inference(image_paths, batch_index):
    model = load_model()
    results = []
    response_times = []
    response_times_path = os.getenv("RESPONSE_TIME_PATH", "response_times.csv")


    # Process each image separately for exact timing
    for img_path in image_paths:
        img = preprocess_image(img_path)
        start_time = time.time()
        with torch.no_grad():
            prediction = model(img)
        end_time = time.time()

        _, predicted_class = torch.max(prediction, 1)
        inference_time = end_time - start_time
        image_name = os.path.basename(img_path)

        # Collect results
        results.append({"image": image_name, "predicted_class": predicted_class.item(), "response_time": inference_time})
        response_times.append(inference_time)

    # try:
    #     pd.DataFrame(response_times, columns=["response_time"]).to_csv(response_times_path, index=False)
    #     print("Response times saved successfully.")
    # except Exception as e:
    #     print(f"Error saving response times: {e}")

    return results

# if __name__ == "__main__":
    
#     image_dir = "mobilenet-imagenet/images/test/"
#     print(f"Running inference on images in directory: {image_dir}")

#     try:
#         inference_results = ray.get(run_inference_on_directory.remote(image_dir))
#         for image_file, prediction in inference_results.items():
#             print(f"Image: {image_file}, Prediction: {prediction['class']}")
#     except Exception as e:
#         print(f"Inference error: {e}")

if __name__ == "__main__":

    image_dir = "mobilenet-imagenet/images/test/"
    all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # ⚡ Split images into batches (batch size can be adjusted)
    batch_size = 20
    image_batches = [all_images[i:i + batch_size] for i in range(0, len(all_images), batch_size)]

    # ⚡ Launch parallel Ray tasks (one per batch)
    futures = [run_batch_inference.remote(batch, idx) for idx, batch in enumerate(image_batches)]
    all_results = ray.get(futures)

    # 📁 Combine all batches into a single CSV
    combined_results = [item for batch_result in all_results for item in batch_result]
    response_times_path = os.getenv("RESPONSE_TIME_PATH", "response_times.csv")
    try:
        pd.DataFrame(combined_results).to_csv(response_times_path, index=False)
        print(f"All batches combined into final_inference_results.csv successfully.")
    except Exception as e:
        print(f"Error saving combined CSV: {e}")