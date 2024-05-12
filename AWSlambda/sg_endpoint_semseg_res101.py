import boto3
import base64
import json
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO

def query_endpoint(input_img):
    endpoint_name = 'jumpstart-dft-mx-semseg-fcn-resnet101-ade'
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-image', Body=input_img, Accept='application/json;verbose')
    return response

def parse_response(query_response):
    response_dict = json.loads(query_response['Body'].read())
    return response_dict['predictions'],response_dict['labels'], response_dict['image_labels']

# with open(img_jpg, 'rb') as file: input_img = file.read()

def generate_segmentation(base64_image_data):
    image_data = base64.b64decode(base64_image_data)
    try:
        query_response = query_endpoint(image_data)
    except Exception as e:
        if e.response['Error']['Code'] == 'ModelError':
            raise Exception(
                 "Backend scripts have been updated in Feb '22 to standardize response "
                 "format of endpoint response."
                 "Previous endpoints may not support verbose response type used in this notebook."
                 f"To use this notebook, please launch the endpoint again. Error: {e}."
            )
        else:
            raise
    try:
        predictions, labels, image_labels =  parse_response(query_response)
    except (TypeError, KeyError) as e:
        raise Exception(
              "Backend scripts have been updated in Feb '22 to standardize response "
              "format of endpoint response."
               "Response from previous endpoints not consistent with this notebook."
               f"To use this notebook, please launch the endpoint again. Error: {e}."
       )
    
    return predictions, labels, image_labels

def filtered_segmentation(predictions, labels, image_labels):
    # Filtered out non-interested segmentation

    ATTENTION_LABELS = ['bench', 'step, stair', 'stairway, staircase','stairs, steps', \
                    'buffet, counter, sideboard', 'stage', 'stool', \
                    'vase', 'plate', 'table', 'shelf', \
                    'desk', 'chest of drawers, chest, bureau, dresser', 'counter', \
                    'case, display case, showcase, vitrine', 'bookcase', 'coffee table, cocktail table',\
                    'countertop', 'kitchen island', 'windowpane, window', 'railing, rail', 'cabinet']
    
    attention_idx = {} # key: class idx, val : pixels tuple
    for img_class in image_labels:
        if img_class in ATTENTION_LABELS:
            img_idx = labels.index(img_class)
            attention_idx[img_idx] = []
    
    npimg = np.array(predictions)
    width, height = npimg.shape
    # assert width == 559 and height==536
    
    original_width = 1600 #Fixed
    original_height = 1069 #Fixed
    
    width_ratio = original_width/width
    height_ratio = original_height/height
    
    for w in range(width):
        for h in range(height):
            if npimg[w,h] in attention_idx.keys():
                attention_idx[int(npimg[w,h])].append((w,h))
    
    width = 1600 #Fixed
    height = 1069 #Fixed
    
    test = Image.new('RGB', (original_width, original_height), color = 'black')
    test_draw = ImageDraw.Draw(test)
    
    offset = int(original_height*0.05)
    for key in attention_idx.keys():
        right_most = max(attention_idx[key])
        left_most = min(attention_idx[key])
        avg_y_idx = int((right_most[1] + left_most[1])/2*(height_ratio))
        shape = [(int(left_most[0]*width_ratio),avg_y_idx-offset),(int(right_most[0]*width_ratio),avg_y_idx-offset)]
        test_draw.line(shape, fill ="white", width = int(original_height*0.10))
    
    # Create a BytesIO object to store the image data
    image_buffer = BytesIO()

    # Save the image to the BytesIO buffer in PNG format
    test.save(image_buffer, format='PNG')

    # Retrieve the image data from the buffer
    image_data = image_buffer.getvalue()

    # Encode the image data to base64
    base64_data = base64.b64encode(image_data).decode('utf-8')
    
    return base64_data
    