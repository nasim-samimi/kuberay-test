import os
import torch
from torchvision import transforms
import torch.quantization
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
# from torchvision.models.detection import ssd_mobilenet_v2, SSDMobileNet_V2_Weights # ssd300_vgg16, SSD300_VGG16_Weights # SSD model in PyTorch
from PIL import Image
import ray
import time

# Ray initialization
ray.init(address='auto')
print("Ray initialized")
print(torch.__version__)

print(torch.backends.quantized.supported_engines)

# Load MobileNet-SSD pre-trained on COCO
# model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)  # SSD model with VGG16 backbone, pre-trained on COCO
# model = ssd_mobilenet_v2(weights=SSDMobileNet_V2_Weights.DEFAULT)
def load_model():
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1)        
    model.eval()  # Set the model to evaluation mode
    return model
# model_ref = ray.put(model)

# Preprocessing function for images
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# Detection function
def detect_objects(image_path):
    # Load and preprocess image with PIL
    model = load_model() 
    start_time_img = time.time()
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    input_tensor = preprocess(img).unsqueeze(0)  # Preprocess and add batch dimension
    end_time_img = time.time()
    start_time_det = time.time()
    with torch.no_grad():  # Inference without tracking gradients
        detections = model(input_tensor)[0]  # Get detections for the image
    end_time_det = time.time()
    # Process detections: extract bounding boxes, labels, and scores
    output = []
    for i in range(len(detections['boxes'])):
        score = detections['scores'][i].item()
        if score > 0.5:  # Apply a confidence threshold
            box = detections['boxes'][i].tolist()  # Bounding box
            label = detections['labels'][i].item()  # Class label
            output.append({"box": box, "label": label, "score": score})
    return {"detections": output, "inference time": end_time_det - start_time_det}
# def detect_objects(image_path):
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
#     model.eval()  # Set the model to evaluation mode
#     img = Image.open(image_path).convert("RGB")

#     # Run inference
#     start_time = time.time()
#     results = model(img)  # YOLOv5 model automatically preprocesses the image
#     end_time = time.time()

#     # Process detections: Convert results to a DataFrame
#     detections = results.pandas().xyxy[0]  # Get bounding boxes, labels, and scores as a DataFrame

#     # Filter and format results based on confidence score threshold
#     output = []
#     for _, row in detections.iterrows():
#         if row['confidence'] > 0.5:  # Apply a confidence threshold
#             box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]  # Bounding box coordinates
#             label = row['name']  # Class label
#             score = row['confidence']  # Confidence score
#             output.append({"box": box, "label": label, "score": score})

#     return {"detections": output, "inference time": end_time - start_time}

@ray.remote
def run_inference_on_directory(image_dir):
    results = {}
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            detections = detect_objects(img_path)  # Run object detection
            results[img_file] = {"detections": detections}
    return results

# Main function to run the job
if __name__ == "__main__":
    image_dir = "mobilenet-ssd-coco/images/"  # Change to your directory containing images
    print("Running inference on images in directory:", image_dir)
    
    inference_results = ray.get(run_inference_on_directory.remote(image_dir))
    
    for image_file, prediction in inference_results.items():
        print(f"Image: {image_file}, Detections: {prediction['detections']}")
