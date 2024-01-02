import gradio as gr
from transformers import pipeline

pipe = pipeline("translation", model="fubuki119/opus-mt-en-hi")

title = "English to Hindi Translator"


description = """
This model is fine-tuned version of <a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-hi">Helsinki-NLP/opus-mt-en-hi</a> model on 
<a href="https://huggingface.co/datasets/cfilt/iitb-english-hindi"> iitb-english-hindi </a> dataset.
<img src="images/app_image.png" width=200px>
"""

def generate(text):
  hi_sen = pipe(text)[0]["translation_text"]
  return hi_sen

input_box = gr.Textbox(label="English sentence", placeholder="You'r english sentence here.", lines=2)
dem = gr.Interface(fn=generate,
                   inputs=input_box,
                   outputs="text",
                   title=title,
                   description=description,
                   examples=[["What a beautiful day"], ["What are you doing?"]])
dem.launch(share=True, debug=True)
