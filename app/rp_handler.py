'''
Contains the handler function that will be called by the serverless.
'''
#%%
import os
import base64
import concurrent.futures

from main_app import pipeline

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA
import torch
torch.cuda.empty_cache()


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    word_level_transcript = job_input['word_level_transcript']
    generation_steps = job_input.get('generation_steps', 40)
    num_images = job_input['image_count_hint']

    results = pipeline(word_level_transcript=word_level_transcript,
                       num_images=num_images,
                       broll_image_steps=generation_steps)

    return results
#%%
from example import example_transcript
job = {
    "input": {
        "word_level_transcript": example_transcript,
        "image_count_hint": 3,
        "generation_steps": 20
    }
}
generate_image(job)


#%%
runpod.serverless.start({"handler": generate_image})
