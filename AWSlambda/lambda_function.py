# import the json utility package since we will be working with a JSON object
import json
# import the AWS SDK (for Python the package name is boto3)
import boto3
# import time 
import time
# import two packages to help us with dates and date formatting
import sg_endpoint_semseg_res101 as semseg
# import stable diffusion model endpoint for sementic segmentation
import sg_endpoint_stdiffusion as stdiff
# import stable diffusion model endpoint to retrive a generated image
import _const
# import constant masking image for testing


# define the handler function that the Lambda service will use as an entry point
def lambda_handler(event, context):

    # extract values from the event object we got from the Lambda service and store in a variable
    base64_input_image = event['origImageRawData']
    userlevel = event['userLevel']
    
    # extract values from the _const for the maksed image and store in a variable
    basae64_mask_image = _const.MASK
    # predictions, labels, image_labels = semseg.generate_segmentation(base64_input_image)
    # basae64_mask_image = semseg.filtered_segmentation(predictions, labels, image_labels)
    
    # return a properly formatted JSON object
    genImageRawData = stdiff.generate_image(base64_input_image,basae64_mask_image,userlevel)
    
    # return a properly formatted JSON object
    result = {'genImageRawData':genImageRawData}
    
    return {
        'statusCode': 200,
        'body': result
    }