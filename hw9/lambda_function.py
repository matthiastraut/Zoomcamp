from io import BytesIO
import os
import numpy as np
import onnxruntime as ort
from urllib import request
from PIL import Image
from torchvision import transforms

onnx_model_path = os.getenv("MODEL_NAME", "hair_classifier_empty.onnx")
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

transform = transforms.Compose([
    # transforms.RandomRotation(50),
    # transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])


def predict(url):
    image = download_image(url)
    image = prepare_image(image, (200, 200))
    image = transform(image)
    image = np.expand_dims(image.cpu().numpy(), 1).reshape((1, 3, 200, 200))
    result = session.run([output_name], {input_name: image})
    prediction = result[0][0].tolist()
    return prediction


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result

# For testing purposes:
# lambda_handler({"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"})