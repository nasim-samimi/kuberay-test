import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import ray

# Ray initialization
ray.init(address='auto')

# Load MobileNetV2 pre-trained model
model = MobileNetV2(weights="imagenet")

# Image inference function
def infer_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=1)[0]
    
    return decoded_preds

# Example function to load and process images in a directory
@ray.remote
def run_inference_on_directory(image_dir):
    results = {}
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if os.path.isfile(img_path):
            predictions = infer_image(img_path)
            results[img_file] = predictions
    return results

# Main function to run the job
if __name__ == "__main__":
    image_dir = "images/"  # Change to your directory containing images
    print("Running inference on images in directory:", image_dir)
    
    inference_results = ray.get(run_inference_on_directory.remote(image_dir))
    
    for image_file, prediction in inference_results.items():
        print(f"Image: {image_file}, Prediction: {prediction}")
