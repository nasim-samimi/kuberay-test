import os
import torch
from torchvision import transforms
import torch.quantization
from torchvision.models.detection import ssd_mobilenet_v2, SSDMobileNet_V2_Weights # ssd300_vgg16, SSD300_VGG16_Weights # SSD model in PyTorch
from PIL import Image
import ray
import time

# Ray initialization
ray.init(address='auto')
print("Ray initialized")

# Load MobileNet-SSD pre-trained on COCO
# model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)  # SSD model with VGG16 backbone, pre-trained on COCO
model = ssd_mobilenet_v2(weights=SSDMobileNet_V2_Weights.COCO_V1)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model.eval()  # Set the model to evaluation mode
model_ref = ray.put(model)

# Preprocessing function for images
preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# Detection function
def detect_objects(image_path, model):
    # Load and preprocess image with PIL
    start_time_img = time.time()
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    input_tensor = preprocess(img).unsqueeze(0)  # Preprocess and add batch dimension
    end_time_img = time.time()
    start_time_det = time.time()
    with torch.no_grad():  # Inference without tracking gradients
        detections = model(input_tensor)[0]  # Get detections for the image
    end_time_det = time.time()
    # Process detections: extract bounding boxes, labels, and scores
    results = []
    for i in range(len(detections['boxes'])):
        score = detections['scores'][i].item()
        if score > 0.5:  # Apply a confidence threshold
            box = detections['boxes'][i].tolist()  # Bounding box
            label = detections['labels'][i].item()  # Class label
            results.append({"box": box, "label": label, "score": score})
    return results

@ray.remote
def run_inference_on_directory(image_dir, model):
    results = {}
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            start_time = time.time()
            detections = detect_objects(img_path, model)  # Run object detection
            end_time = time.time()

            results[img_file] = {"detections": detections, "time": end_time - start_time}
    return results

# Main function to run the job
if __name__ == "__main__":
    image_dir = "mobilenet-ssd-coco/images/"  # Change to your directory containing images
    print("Running inference on images in directory:", image_dir)
    
    inference_results = ray.get(run_inference_on_directory.remote(image_dir,model_ref))
    
    for image_file, prediction in inference_results.items():
        print(f"Image: {image_file}, Detections: {prediction['detections']}, Inference Time: {prediction['time']:.4f} seconds")
