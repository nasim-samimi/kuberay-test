import os
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import ray
import time

# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import (
#     preprocess_input,
#     decode_predictions,
#     MobileNetV2,
# )

# Ray initialization
ray.init(address='auto')
print("Ray initialized")


# Preprocessing function for images
preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    # model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    model.eval()  # Set the model to evaluation mode
    scripted_model = torch.jit.script(model)
    return scripted_model

def preprocess_image(image_path): ## this is for tensorflow
    img = image.load_img(image_path, target_size=(224, 224))  # Load and resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Preprocess input for MobileNetV2

# Image inference function
def infer_image(image_path,model):
    img = Image.open(image_path)
    img = preprocess(img)  # Apply preprocessing
    img = img.unsqueeze(0)  # Add batch dimension
    start_time = time.time()
    with torch.no_grad():  # Inference without tracking gradients
        preds = model(img)
    end_time = time.time()
    # Decode predictions (get top-1 prediction)
    _, predicted_class = torch.max(preds, 1)
    
    return {"detections": predicted_class.item(), "inference time": end_time - start_time}

@ray.remote
def run_inference_on_directory(image_dir):
    model=load_model()
    # model = ray.get(model_ref)  # Retrieve the model from the object store
    results = {}
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            start_time = time.time()
            predicted_class = infer_image(img_path,model) ## this is for pytorch
            # predicted_class, predictions = infer_image_tf(img_path, model) ## this is for tensorflow
            end_time = time.time()

            results[img_file] = {"class": predicted_class, "time": end_time - start_time}
    return results

# Main function to run the job
if __name__ == "__main__":
    image_dir = "mobilenet-imagenet/images/"  # Change to your directory containing images
    print("Running inference on images in directory:", image_dir)
    
    inference_results = ray.get(run_inference_on_directory.remote(image_dir))
    
    for image_file, prediction in inference_results.items():
        print(f"Image: {image_file}, Prediction: {prediction['class']}")
    # print("inference results",inference_results)