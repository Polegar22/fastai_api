import requests
import cv2
import json


FASTAI_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "ugly.jpg"

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread(IMAGE_PATH)
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(FASTAI_REST_API_URL, data=img_encoded.tostring(), headers=headers)

print(json.loads(response.text))