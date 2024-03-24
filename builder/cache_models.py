# builder/model_fetcher.py
#%%
import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
import os
curr_file_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(curr_file_dir, "..", "cache")

# MODEL_ID = "Lykon/dreamshaper-7"
# MODEL_ID = "stablediffusionapi/reliberatev2"
MODEL_ID = "Lykon/dreamshaper-8-lcm"
MODEL_CACHE = cache_dir

def fetch_pretrained_model():
    '''
    Fetch a pretrained model from the Hugging Face model hub.
    '''
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID,
                                                        torch_dtype=torch.float16,
                                                    #   variant="fp16",
                                                    cache_dir=cache_dir,
                                                    load_in_memory=False
    )
    

if __name__ == "__main__":
    fetch_pretrained_model()
#%%