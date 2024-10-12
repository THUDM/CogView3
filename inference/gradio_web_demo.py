"""
THis is the main file for the gradio web demo. It uses the CogView3-Plus-3B model to generate images gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

Usage:
    OpenAI_API_KEY=your_openai_api_key OpenAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

import os
import re
import threading
import time
from datetime import datetime, timedelta

import gradio as gr
import random
from diffusers import CogView3PlusPipeline
import torch
from openai import OpenAI

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.bfloat16).to(device)


def clean_string(s):
    s = s.replace("\n", " ")
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def convert_prompt(
    prompt: str,
    retry_times: int = 5,
) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    system_instruction = """
    You are part of a team of bots that creates images . You work with an assistant bot that will draw anything you say. 
    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an image of a forest morning , as described. 
    You will be prompted by people looking to create detailed , amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. 
    There are a few rules to follow : 
    - Prompt should always be written in English, regardless of the input language. Please provide the prompts in English.
    - You will only ever output a single image description per user request.
    - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting. 
    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.
    - Do not provide the process and explanation, just return the modified English description . Image descriptions must be between 100-200 words. Extra words will be ignored. 
    """

    text = prompt.strip()
    for i in range(retry_times):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{system_instruction}"},
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "一个头发花白的老人"',
                    },
                    {
                        "role": "assistant",
                        "content": "A seasoned male with white hair and a neatly groomed beard stands confidently, donning a dark vest over a striped shirt. His hands are clasped together in front, one adorned with a ring, as he looks directly at the viewer with a composed expression. The soft lighting accentuates his features and the subtle textures of his attire, creating a portrait that exudes sophistication and a timeless elegance.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "画一只老鹰"',
                    },
                    {
                        "role": "assistant",
                        "content": "A majestic eagle with expansive brown and white wings glides through the air, its sharp yellow eyes focused intently ahead. The eagle's talons are poised and ready for hunting, as it soars over a rugged mountainous terrain dusted with snow, under a soft blue sky.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "画一辆摩托车"',
                    },
                    {
                        "role": "assistant",
                        "content": "Parked on a wet city street at night, a sleek motorcycle with a black and green design stands out. Its headlights cast a soft glow, reflecting off the puddles and highlighting its aerodynamic shape. The design is marked by sharp lines and angular features, with gold accents that shine against the dark backdrop. The motorcycle exudes an air of performance and luxury, ready to slice through the urban landscape.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "穿着金色盔甲的人"',
                    },
                    {
                        "role": "assistant",
                        "content": "A figure clad in meticulously crafted, golden armor stands with an air of quiet confidence. The armor, reminiscent of medieval knight attire, features a scalloped design with leaf-like patterns and is complemented by a black, form-fitting undergarment. The helmet, with its angular visor, adds to the intimidating presence. This armor, with its rich gold tones and intricate details, suggests a character of nobility or mythical origin, poised for valorous endeavors.",
                    },
                    {
                        "role": "user",
                        "content": f'Create an imaginative image descriptive caption for the user input : "{text}"',
                    },
                ],
                model="glm-4-plus",
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=300,
            )
            prompt = response.choices[0].message.content
            if prompt:
                prompt = clean_string(prompt)
                break
        except Exception as e:
            pass

    return prompt


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        directories = ["./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()


def infer(
    prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, 65536)

    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    return image, seed


examples = [
    "A vintage pink convertible with glossy chrome finishes and whitewall tires sits parked on an open road, surrounded by a field of wildflowers under a clear blue sky. The car's body is a delicate pastel pink, complementing the vibrant greens and colors of the meadow. Its interior boasts cream leather seats and a polished wooden dashboard, evoking a sense of classic elegance. The sun casts a soft light on the vehicle, highlighting its curves and shiny surfaces, creating a picture of nostalgia mixed with dreamy escapism.",
    "A noble black Labrador retriever sits serenely in a sunlit meadow, its glossy coat absorbing the golden rays of a late afternoon sun. The dog's intelligent eyes sparkle with a mixture of curiosity and loyalty, as it gazes off into the distance where the meadow meets a line of tall, slender birch trees. The dog's posture is regal, yet approachable, with its tongue playfully hanging out to the side slightly, suggesting a friendly disposition. The idyllic setting is filled with the vibrant greens of lush grass and the soft colors of wildflowers speckled throughout, creating a peaceful harmony between the dog and its natural surroundings.",
    "A vibrant red-colored dog of medium build stands attentively in an autumn forest setting. Its fur is a deep, rich red, reminiscent of autumn leaves, contrasting with its bright, intelligent eyes, a clear sky blue. The dog's ears perk up, and its tail wags slightly as it looks off into the distance, its posture suggesting alertness and curiosity. Golden sunlight filters through the canopy of russet and gold leaves above, casting dappled light onto the forest floor and the glossy coat of the canine, creating a serene and heartwarming scene.",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
             CogView3-Plus Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/CogView3-Plus">🤗 Model Hub | 
               <a href="https://github.com/THUDM/CogView3">🌐 Github</a> |
               <a href="https://arxiv.org/abs/2403.05121">📜 arxiv </a>
           </div>
           <div style="text-align: center;display: flex;justify-content: center;align-items: center;margin-top: 1em;margin-bottom: .5em;">
              <span>If the Space is too busy, duplicate it to use privately</span>
              <a href="https://huggingface.co/spaces/THUDM-HF-SPACE/CogView-3-Plus?duplicate=true"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-lg.svg" width="160" style="
                margin-left: .75em;
            "></a>
           </div>
           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. 
            </div>
        """)

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=3,
                placeholder="Enter your prompt",
                container=False,
            )
        with gr.Row():
            enhance = gr.Button("Enhance Prompt (Strongly Suggest)", scale=1)
            enhance.click(convert_prompt, inputs=[prompt], outputs=[prompt])
            run_button = gr.Button("Run", scale=1)
        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=65536,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=2048,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=2048,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=7.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=10,
                    maximum=100,
                    step=1,
                    value=50,
                )

        gr.Examples(examples=examples, inputs=[prompt])
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result, seed],
    )

demo.queue().launch()
