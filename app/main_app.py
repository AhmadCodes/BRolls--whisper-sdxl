# %%
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip
# from moviepy.config import change_settings
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

# change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
import numpy as np
from PIL import Image
import cv2
try:
    # from sdxlturbo.predict_sdxlturbo import Predictor as SDXLPredictor
    from dreamshaper_lcm.predict_DS_LCM import Predictor as SDXLPredictor
except ImportError:

    # from .sdxlturbo.predict_sdxlturbo import Predictor as SDXLPredictor
    from .dreamshaper_lcm.predict_DS_LCM import Predictor as SDXLPredictor



import os

os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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


def validate_KV_pair(dict_list):
    for d in dict_list:
        check_all_keys = all(
            k in d for k in ("description", "start", "end"))
        
        check_description = isinstance(d['description'], str)
        try:
            d['start'] = float(d['start'])
            d['end'] = float(d['end'])
        except:
            return False
        check_start = isinstance(d['start'], float)
        check_end = isinstance(d['end'], float)
        
        return check_all_keys and check_description \
            and check_start and check_end



def json_corrector(json_str, 
                   exception_str,
                   openaiapi_key):
    
    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer {}".format(openaiapi_key)}
    
    prompt_prefix = f"""Exception: {exception_str}
    JSON:{json_str}
    ------------------
    """
    prompt = prompt_prefix + """\n Correct the following JSON, eliminate any formatting issues occured due to misplaces or lack or commas, brackets, semicolons, colons, other symbols, etc
    \nJSON:"""

    # Define the payload for the chat model
    messages = [
        {"role": "system", "content": "You are an expert in correcting JSON strings, you return a VALID JSON by eliminating all formatting issues"},
        {"role": "user", "content": prompt}
    ]
    chatgpt_payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": 2000,
        "top_p": 1,
        "stop": ["###"]
    }
    
    try:
        url = chatgpt_url
        response = requests.post(url, json=chatgpt_payload, headers=headers)
        response_json = response.json()

        try:
            print("response ", response_json['choices'][0]['message']['content'])
        except:
            print("response ", response_json)

            return None
        # Extract data from the API's response
        try:
            output = json.loads(
                response_json['choices'][0]['message']['content'].strip())
            return output
        except Exception as e:
            print("Error in response from OPENAI GPT-3.5: ", e)
            return None
        
    except Exception as e:
        return None

def fetch_broll_description(transcript, wordlevel_info, url, openaiapi_key):
    
    success = False
    err_msg = ""

    if openaiapi_key == "":
        openaiapi_key = OPENAI_API_KEY

    assert openaiapi_key != "", "Please enter your OPENAI API KEY"

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
    Strictly output only JSON in the output using the format (BE CAREFUL NOT TO MISS ANY COMMAS, QUOTES OR SEMICOLONS ETC)-""".format(json.dumps(wordlevel_info), transcript)

    sample = [
        {"description": "...", "start": "...", "end": "..."},
        {"description": "...", "start": "...", "end": "..."}
    ]
    

    prompt = prompt_prefix + json.dumps(sample) + """\nMake the start and end timestamps a minimum duration of more than 3 seconds.
    Also, place them at the appropriate timestamp position where the relevant context is being spoken in the transcript. \nJSON:"""

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
    while not success:
        success = False
        # Make the request to OpenAI's API
        response = requests.post(url, json=chatgpt_payload, headers=headers)
        response_json = response.json()

        try:
            print("response ", response_json['choices'][0]['message']['content'])
        except:
            print("response ", response_json)
            if 'error' in response_json:
                err_msg = response_json['error']
                return None, err_msg
            success = False
            continue
        # Extract data from the API's response
        try:
            output = json.loads(
                response_json['choices'][0]['message']['content'].strip())
            success = validate_KV_pair(output)
            if success:
                print("JSON: ", output)
                success = True
            else:
                print("Could not validate Key-Value pairs in JSON")
                print("Trying again...")
                success = False
                continue
        except Exception as e:
            print("Error in response from OPENAI GPT-4: ", e)
            
            output = json_corrector(response_json['choices'][0]['message']['content'].strip(), 
                                    str(e),
                                    openaiapi_key)
            if output is not None:
                print("Corrected JSON: ", output)
                success = True
            else:
                print("Could not correct JSON")
                print("Trying again...")
                success = False
                continue

    return output, err_msg

# %%


def generate_images(descriptions,
                    steps=3):
    all_images = []

    num_images = len(descriptions)

    negative_prompt = "nsfw, nude, nudity, sexy, naked, ((deformed)), ((limbs cut off)), ((quotes)), ((unrealistic)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs, glitch, low contrast, noisy"

    for i, description in enumerate(descriptions):
        prompt = description['description']

        final_prompt = "((perfect quality)), ((ultrarealistic)), ((realism)) 4k, {}, no occlusion, highly detailed,".format(
            prompt.replace('.', ","))
        img = sdxlpredictor.predict(prompt=final_prompt,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=steps)

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

        start, end = float(item['start']), float(item['end'])
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


def generate_video(all_images, broll_descriptions, wordlevel_info, video_file,
                   progress=None):
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

    progress.progress(0.55, "Overlaying generated clips on the video...")
    if add_subtitles:
        # Generate a list of text clips based on timestamps
        wordclips = [generate_text_clip(
            item['word'], float(item['start']), float(item['end']), video) for item in wordlevel_info]

        # Overlay the text clips on the video
        final_video = CompositeVideoClip([video] + clips + wordclips)
    else:
        final_video = CompositeVideoClip([video] + clips)
    finalvideodir = os.path.dirname(video_file)
    video_file_name = os.path.basename(video_file)
    finalvideoname = video_file_name.split(".")[0] + "_edited.mp4"
    finalvideopath = os.path.join(finalvideodir, finalvideoname)
    print("Saving final video to ", finalvideopath)
    # Write the result to a file
    progress.progress(0.8, "Saving final video...")
    final_video.write_videofile(
        finalvideopath, codec="libx264",
        audio_codec="aac",
        threads=6)
    return finalvideopath

# %%


def pipeline(video_file,
             broll_image_steps=50,
             transcription_model="medium",
             SD_model="lykon/dreamshaper-8-lcm",
             openaiapi_key='',
             progress=None):
    
    if sdxlpredictor.pipe._internal_dict['_name_or_path'] != SD_model:
        sdxlpredictor.setup(model_id=SD_model)
        print("Changed SD model to ", sdxlpredictor.pipe._internal_dict['_name_or_path'])
        
    if progress is not None:
        progress.progress(5, "Extracting Transcript from Video")

    # Extract segments from the audio
    wordlevel_info, transcript = extract_segments_from_audio(video_file,
                                                             model=transcription_model)
    if progress is not None:
        progress.progress(0.15, "Finished Extracting Transcript from Video")
        progress.progress(0.2, "Generating B-roll Image Descriptions")
    # Fetch B-roll descriptions
    broll_descriptions, err_msg = fetch_broll_description(
        transcript, wordlevel_info, chatgpt_url,
        openaiapi_key)
    if err_msg != "" and broll_descriptions is None:
        return None, err_msg
    if progress is not None:
        progress.progress(0.3, "Finished Generating B-roll Image Descriptions")
        progress.progress(0.35, "Generating B-Roll Images")
    # Generate B-roll images
    allimages = generate_images(broll_descriptions,
                                steps=broll_image_steps)
    if progress is not None:
        progress.progress(0.4, "Finished Generating B-roll images")
        progress.progress(0.5, "Compiling Final Video")

    # Generate the final video
    finalvideopath = generate_video(
        allimages, broll_descriptions, wordlevel_info, video_file, 
        progress=progress)
    if progress is not None:
        progress.progress(1, "Finished Editing and Compiling Final Video")

    return finalvideopath, err_msg
# %%


if __name__ == "__main__":
    for i in range(1, 100):
        print("i: ", i)
        video_files = ["/workspace/BRolls--whisper-sdxl/assets/SaaS.mp4",
                       "/workspace/BRolls--whisper-sdxl/assets/basicmp4-download.mp4"]
        for video_file in video_files:
            for sd_model in ["lykon/dreamshaper-8-lcm", "stabilityai/sdxl-turbo", "Lykon/dreamshaper-xl-turbo"]:
                print("SD Model: ", sd_model)
                finalvideopath = pipeline(video_file,
                                          openaiapi_key="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",
                                          SD_model=sd_model)
                print("Final video path: ", finalvideopath)

# %%
