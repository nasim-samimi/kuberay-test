import os
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import ray

# Ray initialization
ray.init(address='auto')
print("Ray initialized")

# Load MobileNetV2 pre-trained model


# model_ref = ray.put(model)

# Preprocessing function for images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image inference function
def infer_image(image_path,model):
    img = Image.open(image_path)
    img = preprocess(img)  # Apply preprocessing
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Inference without tracking gradients
        preds = model(img)

    # Decode predictions (get top-1 prediction)
    _, predicted_class = torch.max(preds, 1)
    
    return predicted_class.item()

@ray.remote
def run_inference_on_directory(image_dir):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.eval()  # Set the model to evaluation mode
    # model = ray.get(model_ref)  # Retrieve the model from the object store
    results = {}
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            predictions = infer_image(img_path,model)
            results[img_file] = predictions
    return results

# Main function to run the job
if __name__ == "__main__":
    image_dir = "images/"  # Change to your directory containing images
    print("Running inference on images in directory:", image_dir)
    
    inference_results = ray.get(run_inference_on_directory.remote(image_dir))
    
    for image_file, prediction in inference_results.items():
        print(f"Image: {image_file}, Prediction: {prediction}")
