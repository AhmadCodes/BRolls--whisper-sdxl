#%%
import os
from typing import List

from diffusers import AutoPipelineForText2Image
import torch


# MODEL_ID = "Lykon/dreamshaper-7"
MODEL_ID = "stabilityai/sdxl-turbo"
MODEL_CACHE = "/workspace/.cache/"
NEGATIVE_PROMPT = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"




CACHE_DIR = "/workspace/.cache/"

class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        # safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        #     SAFETY_MODEL_ID,
        #     cache_dir=MODEL_CACHE,
        #     local_files_only=True,
        # )
        
        self.pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, 
                                                              torch_dtype=torch.float16,
                                                              variant="fp16",
                                                              cache_dir=CACHE_DIR,
                                                              )
        self.pipe.to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()

    @torch.inference_mode()
    def predict(self, prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=3):
        """Run a single prediction on the model"""
        seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")


        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            negative_prompt= NEGATIVE_PROMPT,
            num_inference_steps=num_inference_steps, 
            guidance_scale=0.0
        )

        return output.images[0]


# %%
if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    
#%%
if __name__ == "__main__":

    prompt = "a cat jumping over a dog"
    # prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    output = predictor.predict(prompt, num_inference_steps=3)
    output.save("output.png")

