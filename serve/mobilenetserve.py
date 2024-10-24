import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from ray import serve

@serve.deployment
class ImageClassifier:
    def __init__(self):
        # Load the pre-trained MobileNetV2 model with ImageNet weights
        self.model = MobileNetV2(weights="imagenet")

    async def __call__(self, request):
        # Receive image from the request and prepare it for classification
        form = await request.form()
        img_file = await form["image"].read()

        # Preprocess the image for MobileNetV2 input
        img = image.load_img(img_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get predictions
        preds = self.model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=1)[0]
        return {"predictions": decoded_preds}

# Bind the app to Ray Serve
app = ImageClassifier.bind()
