# %%
import os
from typing import List

from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch
import os
curr_file_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(curr_file_dir, "../..", "cache")

# MODEL_ID = "Lykon/dreamshaper-7"
# MODEL_ID = "stablediffusionapi/reliberatev2"
MODEL_ID = "Lykon/dreamshaper-8-lcm"
MODEL_CACHE = cache_dir
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
NEGATIVE_PROMPT = "((text)), worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
NEGATIVE_PROMPT = "((deformed)), ((limbs cut off)), ((quotes)), ((unrealistic)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs, glitch, low contrast, noisy"

# %%



#%%
CACHE_DIR = cache_dir


class Predictor:
    def setup(self,
              model_id=MODEL_ID,
              safety_model_id=SAFETY_MODEL_ID,
              cache_dir=CACHE_DIR,
              ):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        # safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        #     SAFETY_MODEL_ID,
        #     cache_dir=MODEL_CACHE,
        #     local_files_only=True,
        # )

        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id,
                                                              torch_dtype=torch.float16,
                                                            #   variant="fp16",
                                                            cache_dir=cache_dir,
                                                              )
        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config)

        self.pipe.to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_model_cpu_offload()

    @torch.inference_mode()
    def predict(self, prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=3,
                guidance_scale=1.0,):
        """Run a single prediction on the model"""
        seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        return output.images[0]


# %%
if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

# %%
if __name__ == "__main__":

    dummies = [{"description": "A bright, highlighted light bulb signifying a new idea", "start": 1.4, "end": 5.0},
               {"description": "Tools related to a teacher's work, like a report card, quiz paper, and marker pen",
                   "start": 6.3, "end": 9.86},
               {"description": "Growing chart upward symbolizing increase, coined with a lot of people signifying virality",
                "start": 13.38, "end": 16.6},
               {"description": "A large crowd of diverse people walking along a city street",
                "start": 18.18, "end": 20.78},
               {"description": "An ingenious invention partially built, including various associated tools around suggesting building approx",
                "start": 24.1, "end": 26.4},
               {"description": "Intersected movie clapperboard with genre names as codes",
                "start": 28.34, "end": 31.48},
               {"description": "Decoration set around the creating process involved in the highest-grossing category - an action film",
                "start": 34.32, "end": 35.6},
               {"description": "Collections of different types of genres-symbolizing objects",
                "start": 40.85, "end": 44.5},
               {"description": "Flat perspective high-infrastructure action scene illustration",
                "start": 50.4, "end": 52.2},
               {"description": "Shoot of extreme large audience arriving for a laser up night claiming huge target audience", "start": 57.78, "end": 59.1}]

    for dummy in dummies:
        NEGATIVE_PROMPT = "((text)), ((limbs cut off)), ((quotes)), ((unrealistic)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs, glitch, low contrast, noisy"

        prompt = dummy["description"]
        guidance_scale = 0.0
        output = predictor.predict(prompt, num_inference_steps=50,
                                   negative_prompt=NEGATIVE_PROMPT,
                                   guidance_scale=guidance_scale)
        if not os.path.exists("./static"):
            os.makedirs("./static")
        output.save(f"./static/output_DSLCM_{dummy['start']}_{dummy['end']}_GS{guidance_scale}.png")


# %%
