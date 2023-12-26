#%%
import os
from typing import List

import torch
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
# from diffusers.pipelines.stable_diffusion.safety_checker import (
#     StableDiffusionSafetyChecker,
# )


MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "/workspace/.cache/"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
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

        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            # safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    def predict(self, prompt, num_inference_steps=35):
        """Run a single prediction on the model"""
        seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")


        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            negative_prompt= NEGATIVE_PROMPT,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        return output.images[0]


# %%
if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    
#%%
if __name__ == "__main__":

    prompt = "a photo of a cat"
    output = predictor.predict(prompt)
    output.save("output.png")
