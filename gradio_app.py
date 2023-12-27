
#%%
import gradio as gr
from app.main_app import pipeline
#%%
model_options = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]

def generate_video(video_file, apikey='', model_type="medium"):
    if apikey == "":
        return "Please enter your OPENAI API KEY"
    finalvideopath = pipeline(video_file,
                              model_type,
                              apikey
                              )
    return finalvideopath

interface = gr.Interface(
    fn=generate_video, 
    inputs=[gr.Video( sources=["upload"],
                     label="Input Video"),
            gr.Text(label="OPENAI API KEY"),
            gr.Dropdown(choices=model_options, label="""Model Type (Default: Medium)""",)], 
    outputs=gr.Video(),
    title="B-Roll Images",
    description="Generate B-roll Images and insert to video",
    allow_flagging = False,
    css="footer {visibility: hidden}"
)

interface.launch()


# %%
