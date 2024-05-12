import boto3
import json
import base64

# Define constant variable
ENDPOINT_NAME = 'jumpstart-dft-stable-diffusion-2-inpainting'
LEVEL_DICT = {"... (Basic)" : ["Prickly Pear Cactus", "Red Pancake", "Golden Barrel Cactus", "mesquite", "creosote bush", "yucca", "cactus"]
                , "I am good. (Intermediate)": ["peace lilies", "ZZ plants", "philodendrons"]
                , "I am Groot! (Advanced)": ["bonsai trees", "colorful orchids", "Venus flytraps"]}

PLANTSTYLE = ["None", "Tropical Plants", "Coastal Plants", "Aquatic plants"]

def encode_img(img_name):
    with open(img_name,'rb') as f: img_bytes = f.read()
    encoded_img = base64.b64encode(bytearray(img_bytes)).decode()
    return encoded_img

def query_endpoint(payload):
    """query the endpoint with the json payload encoded in utf-8 format."""
    encoded_payload = json.dumps(payload).encode('utf-8')
    client = boto3.client('runtime.sagemaker')
    # Accept = 'application/json;jpeg' returns the jpeg image as bytes encoded by base64.b64 encoding.
    # To receive raw image with rgb value set Accept = 'application/json'
    # To send raw image, you can set content_type = 'application/json' and encoded_image as np.array(PIL.Image.open('low_res_image.jpg')).tolist()
    # Note that sending or receiving payload with raw/rgb values may hit default limits for the input payload and the response size.
    response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json;jpeg', Accept = 'application/json;jpeg', Body=encoded_payload)
    return response


def generate_image(base64_input_image, basae64_mask_image, userlevel):

    prompt = f"mainly consist of {PLANTSTYLE[0]}, {LEVEL_DICT[userlevel]} are each arranged in their own beautiful flower pots on the shelf, natural beauty, eco-friendliness, \
    incredibly detailed, 8k resolution, photo-realistic, cosy, warm, and comfortable environment of the room, with ample natural light"
                
    payload = { "prompt": prompt 
               ,"image": base64_input_image
               ,"mask_image":basae64_mask_image
               ,"num_inference_steps":50
               , "guidance_scale":7.5, "seed": 1}
    
    query_response = query_endpoint(payload)
    
    response_dict = json.loads(query_response['Body'].read())
    generated_images = response_dict['generated_images']
    
    return generated_images[0]

