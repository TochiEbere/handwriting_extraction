from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

import os
import time
import uvicorn
from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel

subscription_key = "********"
endpoint = "https://ocr-handwriting-extraction.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

app= FastAPI()

@app.get('/')
def welcome():
    return 'Welcome to the handwriting extraction app'
    

@app.post('/extract_text')
async def predict_(Image: UploadFile = File(...)):
    
    # Call API with image and raw response (allows you to get the operation location)
    recognize_handwriting_results = computervision_client.read_in_stream(Image.file, raw=True)
    # Get the operation location (URL with ID as last appendage)
    operation_location_local = recognize_handwriting_results.headers["Operation-Location"]
    # Take the ID off and use to get results
    operation_id_local = operation_location_local.split("/")[-1]


    while True:
        recognize_handwriting_result = computervision_client.get_read_result(operation_id_local)
        if recognize_handwriting_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    sentences = []
    if recognize_handwriting_result.status == OperationStatusCodes.succeeded:
        for text_result in recognize_handwriting_result.analyze_result.read_results:
            for line in text_result.lines:
                sentences.append(line.text)
    
    return [line for line in sentences]

# if __name__=="__main__":
#     uvicorn.run(app, host='127.0.0.1', port=8000)
