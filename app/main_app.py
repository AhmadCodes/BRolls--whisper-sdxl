# %%

# from moviepy.config import change_settings
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

# change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
import uuid
from io import BytesIO
import boto3
from dotenv import load_dotenv
import requests
import json

try:
    # from sdxlturbo.predict_sdxlturbo import Predictor as SDXLPredictor
    from dreamshaper_lcm.predict_DS_LCM import Predictor as SDXLPredictor
except ImportError:

    # from .sdxlturbo.predict_sdxlturbo import Predictor as SDXLPredictor
    from .dreamshaper_lcm.predict_DS_LCM import Predictor as SDXLPredictor


import os

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), "..", '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
def get_s3_client():
    s3_client = boto3.client('s3',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return s3_client


# %%

sdxlpredictor = SDXLPredictor()
sdxlpredictor.setup()

S3_CLIENT = get_s3_client()
# %%




def chunk_level_transcript(word_level_transcript, chunk_size_s=3):
    chunked_ts = []
    start_t = 0
    started = False
    end_t = 0
    chunk = []
    
    for i in range(0, len(word_level_transcript)):
        chunk.append(word_level_transcript[i])
        if not started:
            start_t = float(word_level_transcript[i]['start'])
            started = True
        end_t = float(word_level_transcript[i]['end'])
        delta = end_t - start_t
        if delta >= chunk_size_s:
            chunk_words_str = " ".join([w['word'] for w in chunk])
            chunk_start = chunk[0]['start']
            chunk_end = chunk[-1]['end']
            chunked_ts.append(
                {"start": chunk_start, "end": chunk_end, "segment": chunk_words_str})
            started = False
            chunk = []
            
    if len(chunk) > 0:
        chunk_words_str = " ".join([w['word'] for w in chunk])
        chunk_start = chunk[0]['start']
        chunk_end = chunk[-1]['end']
        chunked_ts.append(
            {"start": chunk_start, "end": chunk_end, "segment": chunk_words_str})
    return chunked_ts

def convert_to_srt(word_level_transcript):
    srt = ""
    for i, word in enumerate(word_level_transcript):
        srt += f"{i + 1}\n"
        srt += f"{word['start']} --> {word['end']}\n"
        srt += f"{word['segment']}\n\n"
    return srt



# %%
chatgpt_url = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%
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
            print("response ",
                  response_json['choices'][0]['message']['content'])
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


def fetch_broll_description(wordlevel_info,
                            num_images,
                            url,
                            openaiapi_key):

    success = False
    err_msg = ""

    if openaiapi_key == "":
        openaiapi_key = OPENAI_API_KEY

    assert openaiapi_key != "", "Please enter your OPENAI API KEY"

    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer {}".format(openaiapi_key)}

    chunklevelinfo = chunk_level_transcript(wordlevel_info, chunk_size_s=5)
    subtitles = convert_to_srt(chunklevelinfo)

    prompt_prefix = """{}
    
    Given the subtitles of a video, generate very relevant stock image descriptions to insert as B-roll images.
    The start and end timestamps of the B-roll images should perfectly match with the content that is spoken at that time.
    Strictly don't include any exact word or text labels to be depicted in the images.
    Don't make the timestamps of different illustrations overlap.
    Leave enough time gap between different B-Roll image appearances so that the original footage is also played as necessary.
    Strictly output only JSON in the output using the format (BE CAREFUL NOT TO MISS ANY COMMAS, QUOTES OR SEMICOLONS ETC)-""".format(subtitles)

    sample = [
        {"description": "...", "start": "...", "end": "..."},
        {"description": "...", "start": "...", "end": "..."}
    ]

    prompt = prompt_prefix + json.dumps(sample) + f"""\nMake the start and end timestamps a minimum duration of more than 3 seconds.
    Also, place them at the appropriate timestamp position where the relevant context is being spoken in the transcript. 
    Be sure to only make {num_images} jsons. \nJSON:"""

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
            print("response ",
                  response_json['choices'][0]['message']['content'])
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


def upload_image_to_s3(image, bucket, s3_client):
    # Convert PIL Image to Bytes
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_byte_array = buffer.getvalue()
    # uuid for uploaded_image as the image name
    upload_key = uuid.uuid4().hex + ".png"
    # Upload Bytes to S3
    s3_client.put_object(Body=image_byte_array, Bucket=bucket, Key=upload_key)

    # Generate the URL of the uploaded image
    s3_img_info = {
        "bucket": bucket,
        "key": upload_key
    }

    return s3_img_info

# %%


def pipeline(word_level_transcript,
             num_images=3,
             broll_image_steps=50,
             openaiapi_key=os.getenv("OPENAI_API_KEY"),
             ):


    # Fetch B-roll descriptions
    broll_descriptions, err_msg = fetch_broll_description(word_level_transcript,
                                                          num_images,
                                                          chatgpt_url,
                                                          openaiapi_key)
    if err_msg != "" and broll_descriptions is None:
        return err_msg

    # Generate B-roll images
    allimages = generate_images(broll_descriptions,
                                steps=broll_image_steps)
    img_upload_info = []
    for i, img in enumerate(allimages):
        img_info = upload_image_to_s3(img, "brollimages", S3_CLIENT)
        img_info['description'] = broll_descriptions[i]['description']
        img_info['start'] = broll_descriptions[i]['start']
        img_info['end'] = broll_descriptions[i]['end']
        img_upload_info.append(img_info)

    return img_upload_info
# %%


if __name__ == "__main__":
    from example import example_transcript

    img_info = pipeline(example_transcript,
                        num_images=3,
                        broll_image_steps=50,
                        SD_model="lykon/dreamshaper-8-lcm",
                        openaiapi_key=OPENAI_API_KEY)
# %%
