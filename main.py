import tensorflow
import numpy as np
from tensorflow.keras.preprocessing import image
from google.cloud import storage
from PIL import Image
import urllib.request
# import boto3
# import os

# ACCESS_KEY = os.environ.get('ACCESS_KEY')
# SECRET_KEY = os.environ.get('SECRET_KEY')

score_model = None
binary_model = None
BUCKET_NAME = 'tensorflow2test'
# S3_BUCKET_NAME = 'chana_s3'

# def download_from_s3(bucket_name, object_name, file_name):
#     s3 = boto3.client(
#         's3',
#         aws_access_key_id=ACCESS_KEY,
#         aws_secret_access_key=SECRET_KEY
#         )
#     s3.download_file(bucket_name, object_name, file_name)

def download_from_gcs(bucket_name, object_name, file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    obj = bucket.blob(object_name)
    obj.download_to_filename(file_name)

def http(request):
    global score_model
    global binary_model
    if score_model is None:
        download_from_gcs(BUCKET_NAME, 'resnet50v2_model.h5', '/tmp/score_model.h5')
        score_model = tensorflow.keras.models.load_model('/tmp/score_model.h5')
    if binary_model is None:
        download_from_gcs(BUCKET_NAME, 'tooth_model_binary.h5', '/tmp/binary_model.h5')
        binary_model = tensorflow.keras.models.load_model('/tmp/binary_model.h5')    
    request_json = request.get_json(silent=True)
    image_uri = request_json['uri']
    image_name = image_uri.split('/')[-1]
    image_data = urllib.request.urlopen(image_uri).read()
    with open('/tmp/'+image_name, 'wb') as f:
        f.write(image_data)
        print('image saved')
    
    binary = predict(binary_model, '/tmp/'+image_name, (200,100))
    if binary < 0.5:
        isOrth = 'NO'
    else:
        isOrth = 'YES'
    score = predict(score_model, '/tmp/'+image_name, (224,224))
    result = {'braces': isOrth, 'score' : str(score)}
    return result

def predict(model, image_local_path, targetsize):
    img = image.load_img(image_local_path, target_size=targetsize)
    imgarr = image.img_to_array(img)
    imgarr = imgarr / 255.
    pred = model.predict(np.array([imgarr]))[0][0]
    return pred

