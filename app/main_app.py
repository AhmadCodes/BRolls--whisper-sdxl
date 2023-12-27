# %%
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip
import numpy as np
from PIL import Image
import cv2
try:
    from sdxlturbo.predict_sdxlturbo import Predictor as SDXLPredictor
except ImportError:

    from .sdxlturbo.predict_sdxlturbo import Predictor as SDXLPredictor
        
        

from pprint import pprint
import json
import requests
import ffmpeg
from dotenv import load_dotenv
try:
    from whisper_model.predict_whisper import Predictor as WhisperPredictor
except ImportError:
    from .whisper_model.predict_whisper import Predictor as WhisperPredictor
import os

# %%
whisper_model = WhisperPredictor()
whisper_model.setup()

sdxlpredictor = SDXLPredictor()
sdxlpredictor.setup()


# %%



# %%
chatgpt_url = "https://api.openai.com/v1/chat/completions"


# %%


def extract_segments_from_audio(video_file, model="medium"):

    prediction = whisper_model.predict(video_file, model_name=model)
    wordlevel_info = []
    for item in prediction['segments']:
        words = item.get('words', [])
        for word_info in words:
            wordlevel_info.append({
                'word': word_info.get('word', ''),
                'start': word_info.get('start', 0.0),
                'end': word_info.get('end', 0.0)
            })

    transcript = prediction['transcription']
    return wordlevel_info, transcript
# %%


def fetch_broll_description(transcript, wordlevel_info, url, openaiapi_key):

    headers = {
    "content-type": "application/json",
    "Authorization": "Bearer {}".format(openaiapi_key)}
    
    prompt_prefix = """{}
    transcript: {}
    ------------------
    Given this transcript and corresponding word-level timestamp information, generate very relevant stock image descriptions to insert as B-roll images.
    The start and end timestamps of the B-roll images should perfectly match with the content that is spoken at that time.
    Strictly don't include any exact word or text labels to be depicted in the images.
    Don't make the timestamps of different illustrations overlap.
    Leave enough time gap between different B-Roll image appearances so that the original footage is also played as necessary.
    Strictly output only JSON in the output using the format-""".format(json.dumps(wordlevel_info), transcript)

    sample = [
        {"description": "...", "start": "...", "end": "..."},
        {"description": "...", "start": "...", "end": "..."}
    ]

    prompt = prompt_prefix + json.dumps(sample) + """\nMake the start and end timestamps a minimum duration of more than 3 seconds.
    Also, place them at the appropriate timestamp position where the relevant context is being spoken in the transcript.\nJSON:"""

    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "You are an expert short form video script writer for Instagram Reels and Youtube shorts."},
        {"role": "user", "content": prompt}
    ]

    chatgpt_payload = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 2000,
        "top_p": 1,
        "stop": ["###"]
    }

    # Make the request to OpenAI's API
    response = requests.post(url, json=chatgpt_payload, headers=headers)
    response_json = response.json()

    print("response ", response_json['choices'][0]['message']['content'])

    # Extract data from the API's response
    output = json.loads(
        response_json['choices'][0]['message']['content'].strip())
    print("output ", output)

    return output

# %%




def generate_images(descriptions):
    all_images = []

    num_images = len(descriptions)

    negative_prompt = "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs, glitch, low contrast, noisy"

    for i, description in enumerate(descriptions):
        prompt = description['description']

        final_prompt = "((perfect quality)), 4k, {}, no occlusion, highly detailed,".format(
            prompt.replace('.', ","))
        img = sdxlpredictor.predict(prompt=final_prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=3)

        print(f"Image {i + 1}/{num_images} is generated")
        # img will be a PIL image
        all_images.append(img)

    return all_images


# %%


def create_combined_clips(allimages, b_roll_descriptions, output_resolution=(1080, 1920), fps=24):
    video_clips = []

    # Iterate over the images and descriptions
    for img, item in zip(allimages, b_roll_descriptions):
        img = np.array(img)
        img_resized = cv2.resize(
            img, (output_resolution[0], output_resolution[0]))

        start, end = item['start'], item['end']
        duration = end-start

        # Blur the image
        blurred_img = cv2.GaussianBlur(img, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)
        blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)

        # Overlay the original image on the blurred one
        y_offset = (output_resolution[1] - output_resolution[0]) // 2
        blurred_img[y_offset:y_offset+output_resolution[0], :] = img_resized

        cv2.imwrite("test_blurred_image.jpg", blurred_img)

        video_clip = ImageClip(np.array(blurred_img)).with_position(
            'center').with_duration(end - start)
        video_clip = video_clip.with_start(start)
        video_clips.append(video_clip)

    return video_clips

# %%


def generate_video(all_images, broll_descriptions, wordlevel_info, video_file):
    # Function to generate text clips
    def generate_text_clip(word, start, end, video):
        txt_clip = (TextClip(word, font_size=80, color='white', font="Nimbus-Sans-Bold", stroke_width=3, stroke_color='black').with_position("center")
                    .with_duration(end - start))

        return txt_clip.with_start(start)

    # Load the video file
    video = VideoFileClip(video_file)

    print(video.size)

    # Generate a list of text clips based on timestamps
    clips = create_combined_clips(
        all_images, broll_descriptions, output_resolution=video.size, fps=24)

    add_subtitles = True

    if add_subtitles:
        # Generate a list of text clips based on timestamps
        wordclips = [generate_text_clip(
            item['word'], item['start'], item['end'], video) for item in wordlevel_info]

        # Overlay the text clips on the video
        final_video = CompositeVideoClip([video] + clips + wordclips)
    else:
        final_video = CompositeVideoClip([video] + clips)
    finalvideodir = os.path.dirname(video_file)

    finalvideoname = "final.mp4"
    finalvideopath = os.path.join(finalvideodir, finalvideoname)
    # Write the result to a file
    final_video.write_videofile(
        finalvideopath, codec="libx264", audio_codec="aac")
    return finalvideopath

# %%


def pipeline(video_file, transcription_model="medium",
             openaiapi_key=''):

    # Extract segments from the audio
    wordlevel_info, transcript = extract_segments_from_audio(video_file,
                                                             model=transcription_model)

    # Fetch B-roll descriptions
    broll_descriptions = fetch_broll_description(
        transcript, wordlevel_info, chatgpt_url,
        openaiapi_key)

    # Generate B-roll images
    allimages = generate_images(broll_descriptions)

    # Generate the final video
    finalvideopath = generate_video(
        allimages, broll_descriptions, wordlevel_info, video_file)

    return finalvideopath
# %%


if __name__ == "__main__":
    video_file = "/workspace/BRolls--whisper-sdxl/assets/SaaS.mp4"
    finalvideopath = pipeline(video_file)
    print("Final video path: ", finalvideopath)

# %%
