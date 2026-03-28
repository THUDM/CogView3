“””
This script demonstrates how to generate an image using the CogView4-6B model within the Hugging Face Space interface. Simply interact with the Gradio interface hosted on Hugging Face CogView4 Demo at [CogView4-6B Hugging Face Space](https://huggingface.co/spaces/THUDM-HF-SPACE/CogView4)

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
OPENAI_API_KEY=”your ZhipuAI API keys” OPENAI_BASE_URL=”https://open.bigmodel.cn/api/paas/v4” python gradio_web_demo.py
```

We use [glm-4-plus](https://bigmodel.cn/dev/howuse/glm-4) as the large model for prompt refinement. You can also choose other large models, such as GPT-4o or MiniMax-M2.7, for refinement.”

### Using MiniMax for prompt enhancement:
```bash
MINIMAX_API_KEY=”your MiniMax API key” LLM_PROVIDER=minimax python gradio_web_demo.py
```

For Different GPU Memory Usage:

12G VRAM
```
MODE=1 OPENAI_API_KEY=”your ZhipuAI API keys” OPENAI_BASE_URL=”https://open.bigmodel.cn/api/paas/v4” python gradio_web_demo.py
```
24G VRAM 32G RAM
```
MODE=2 OPENAI_API_KEY=”your ZhipuAI API keys” OPENAI_BASE_URL=”https://open.bigmodel.cn/api/paas/v4” python gradio_web_demo.py
```
24G VRAM 64G RAM
```
MODE=3 OPENAI_API_KEY=”your ZhipuAI API keys” OPENAI_BASE_URL=”https://open.bigmodel.cn/api/paas/v4” python gradio_web_demo.py
```
40G VRAM 64G RAM and Larger
```
OPENAI_API_KEY=”your ZhipuAI API keys” OPENAI_BASE_URL=”https://open.bigmodel.cn/api/paas/v4” python gradio_web_demo.py
```
“””

import gc
import os
import random
import re
import threading
import time
from datetime import datetime, timedelta

import gradio as gr
import torch
from diffusers import CogView4Pipeline
from diffusers.models import CogView4Transformer2DModel
from openai import OpenAI
from torchao.quantization import int8_weight_only, quantize_
from transformers import GlmModel

from prompt_optimize import PROVIDER_PRESETS


total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824

print(f"\033[32mCUDA版本：{torch.version.cuda}\033[0m")
print(f"\033[32mPytorch版本：{torch.__version__}\033[0m")
print(f"\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m")
print(f"\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m")

if torch.cuda.get_device_capability()[0] >= 8:
    print("\033[32m支持BF16\033[0m")
    dtype = torch.bfloat16
else:
    print("\033[32m不支持BF16，使用FP16\033[0m")
    dtype = torch.float16

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "THUDM/CogView4-6B"
mode = os.environ.get("MODE", "0")
llm_provider = os.environ.get("LLM_PROVIDER", "zhipu")

text_encoder = None
transformer = None
if mode in ["1", "2"]:
    text_encoder = GlmModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
    transformer = CogView4Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    quantize_(text_encoder, int8_weight_only())
    quantize_(transformer, int8_weight_only())

pipe = CogView4Pipeline.from_pretrained(
    model_path,
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=dtype,
).to(device)

if mode in ["1", "3"]:
    pipe.enable_model_cpu_offload()

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


def clean_string(s):
    s = s.replace("\n", " ")
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def convert_prompt(
    prompt: str,
    key: str,
    provider: str = "zhipu",
    retry_times: int = 5,
) -> str:
    # Resolve provider preset
    preset_base_url, preset_model = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["zhipu"])

    # For zhipu provider, honor OPENAI_BASE_URL env var for backwards compatibility
    if provider == "zhipu":
        base_url = os.environ.get("OPENAI_BASE_URL", preset_base_url)
        model = "glm-4-flash"
    else:
        base_url = preset_base_url
        model = preset_model

    # Resolve API key
    if key:
        api_key = key
    elif provider == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        return prompt

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = clean_string(prompt)

    # MiniMax requires temperature in (0.0, 1.0]
    temperature = 0.01
    if "minimax" in base_url.lower():
        temperature = max(temperature, 0.01)

    for i in range(retry_times):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": 'You are a bilingual image description assistant that works with an image generation bot.  You work with an assistant bot that will draw anything you say . \n    For example ,For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" or "阳光透过树林的美丽清晨" will trigger your partner bot to output an image of a forest morning, as described . \n    You will be prompted by people looking to create detailed , amazing images . The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive . \n    There are a few rules to follow : \n    - Input can be in Chinese or English. If input is in English, prompt should be written in English. If input is in Chinese, prompt should be written in Chinese.\n    - You will only ever output a single image description per user request .\n    - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting. \n    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.\n    - Do not provide the process and explanation, just return the modified description . \n    ',
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "An anime girl stands amidst a dense flower bush."',
                    },
                    {
                        "role": "assistant",
                        "content": "This image is a beautifully crafted digital illustration in an anime style. It features Suzuran, a character from the video game Arknights. She stands gracefully amidst a picturesque meadow with lush green grass and scattered wildflowers that gently sway in the breeze. Her attire includes a detailed outfit with layered ruffles and intricate fastenings, reflecting both elegance and functionality typical of the game's aesthetic. Suzuran holds a small bouquet of flowers delicately in her hands, adding to the serene atmosphere.\n\nProminent fox ears extend from her head, signifying her animal spirit form or persona within the game. The background showcases rolling hills covered in dense foliage under a brilliant blue sky dotted with fluffy white clouds. Golden leaves float whimsically through the air, enhancing the magical quality of the scene. A distant building partially obscured by vegetation hints at human presence without detracting from the natural beauty surrounding Suzuran. Overall, this artwork captures a moment of tranquility and charm, embodying the essence of the character and the enchanting world she inhabits.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "Draw a bright convertible car with a sense of artistic design."',
                    },
                    {
                        "role": "assistant",
                        "content": 'The image showcases a meticulously crafted Delahaye 135 M.S. Special Roadster from 1937 in a highly polished and realistic rendering style that highlights its luxurious design and impeccable details. The car\'s body is an elegant deep brown with a glossy finish, exuding sophistication and timeless beauty. Its aerodynamic, streamlined shape features smooth curves accentuated by chrome detailing on the fenders and running boards. The front grille has three prominent circular headlights, adding to its classic charm while the dual exhaust pipes are tastefully integrated into the rear fenders.\n\nThe open cockpit area reveals sumptuous tan leather seats, emphasizing both comfort and elegance for the driver and passenger. The vehicle’s number plate reads "MB 175". Photographed against a dark gradient background, the focus remains solely on this automotive masterpiece, allowing viewers to appreciate every intricate detail without distraction. This image captures not just a car but a piece of art, reflecting the luxury and craftsmanship emblematic of pre-war European automobiles.',
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "画一个白发、神情忧郁的动漫女孩，手里拿着一支香烟"',
                    },
                    {
                        "role": "assistant",
                        "content": "这幅图像是一幅动漫风格的插画，画中描绘了一位长发飘逸的白发女孩。她神情忧郁，双眼低垂，脸上带着微微的愁容。女孩穿着浅色外套，里面搭配深色衬衫和领带，增添了她沉静却时尚的外表。她的手靠近嘴边，似乎在拿着一支香烟，画作的说明也表示她在吸烟。背景描绘了一个冬季的城市场景，地面和建筑物上覆盖着积雪。几株稀疏的植物从雪地中探出，街边停着一辆车，似乎是空荡荡的街道，增强了场景的荒凉氛围。这幅作品整体风格细致精美，典型的现代动漫插画风格，既捕捉了角色的情感表达，也呈现了周围环境的静谧氛围",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "一张红色的海报，中间写有“开门大吉”"',
                    },
                    {
                        "role": "assistant",
                        "content": "这张图片采用了海报风格，色彩鲜艳，主要以红色和金色为主，寓意吉祥如意。在画面的正中间是一块红色的大匾，上面用白色大字写着“开门大吉”，四角点缀着金色的装饰图案，显得格外喜庆。匾额上方悬挂着一对红色的灯笼，增添了节日的气氛。背景左右两侧是传统的绿色中式建筑，屋顶呈现出典型的飞檐翘角设计。底部有祥云朵朵和可爱的卡通福袋，象征着好运福气满满。整张海报传达出浓厚的节日氛围。",
                    },
                    {
                        "role": "user",
                        "content": f"Create an imaginative image descriptive caption for the user input : {prompt}",
                    },
                ],
                model=model,
                temperature=temperature,
                top_p=0.7,
                stream=False,
                max_tokens=300,
            )
            prompt = response.choices[0].message.content
            if prompt:
                prompt = clean_string(prompt)
                break
        except Exception:
            pass

    return prompt


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        os.makedirs("./gradio_tmp", exist_ok=True)
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
    num_images,
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if randomize_seed:
        seed = random.randint(0, 65536)

    images = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=torch.Generator().manual_seed(seed),
    ).images

    return images, seed


def update_max_height(width):
    max_height = MAX_PIXELS // width
    return gr.update(maximum=max_height)


def update_max_width(height):
    max_width = MAX_PIXELS // height
    return gr.update(maximum=max_width)


examples = [
    "这是一幅充满皮克斯风格的动画渲染图像，展现了一只拟人化的粘土风格小蛇。这条快乐的小蛇身着魔术师装扮，占据了画面下方三分之一的位置，显得俏皮而生动。它的头上戴着一顶黑色羊毛材质的复古礼帽，身上穿着一件设计独特的红色棉袄，白色的毛袖增添了一抹温暖的对比。小蛇的鳞片上精心绘制了金色梅花花纹，显得既华丽又不失可爱。它的腹部和脸庞呈现洁白，与红色的身体形成鲜明对比。 这条蜿蜒的小蛇拥有可爱的塑胶手办质感，仿佛随时会从画面中跃然而出。背景是一片鲜艳的红色，地面上散布着宝箱、金蛋和红色灯笼等装饰物，营造出浓厚的节日气氛。画面的上半部分用金色连体字书写着 “Happy New Year”，庆祝新年的到来，同时也暗示了蛇年的到来，为整幅画面增添了一份节日的喜悦和祥瑞。",
    "在这幅如梦似幻的画作中，一辆由云朵构成的毛绒汽车轻盈地漂浮在蔚蓝的高空之中。这辆汽车设计独特，车身完全由洁白、蓬松的云朵编织而成，每一处都散发着柔软而毛茸茸的质感。从车顶到轮胎，再到它的圆润车灯，无一不是由细腻的云丝构成，仿佛随时都可能随风轻轻摆动。车窗也是由透明的云物质构成，同样覆盖着一层细软的绒毛，让人不禁想要触摸。 这辆神奇的云朵汽车仿佛是魔法世界中的交通工具，它悬浮在夕阳映照的绚丽天空之中，周围是五彩斑斓的晚霞和悠然飘浮的云彩。夕阳的余晖洒在云朵车上，为其柔软的轮廓镀上了一层金色的光辉，使得整个场景既温馨又神秘，引人入胜。",
    "A vintage red convertible with gleaming chrome finishes sits attractively under the golden hues of a setting sun, parked on a deserted cobblestone street in a charming old town. The car's polished body reflects the surrounding quaint buildings and the few early evening stars beginning to twinkle in the gentle gradient of the twilight sky. A light breeze teases the few fallen leaves near the car's pristine white-walled tires, which rest casually by the sidewalk, hinting at the leisurely pace of life in this serene setting.",
]
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">CogView4-6B</h2>
            </div>
            <div style="text-align: center;">
                <a href="https://github.com/THUDM/CogView4">🌐 Github</a> |
                <a href="https://arxiv.org/abs/2403.05121">📜 arXiv </a>
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            </div>
        """)

    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=15,
                        placeholder="Enter your prompt",
                        container=False,
                    )
                with gr.Row():
                    enhance = gr.Button("Enhance Prompt (Strongly Suggest)", scale=1)
                    run_button = gr.Button("Run", scale=1)
                with gr.Row():
                    num_images = gr.Number(
                        label="Number of Images",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=2,
                    )
                    key = gr.Textbox(
                        label="Key",
                        placeholder="Enter your key",
                        type="password",
                        max_lines=1,
                    )
                    provider = gr.Dropdown(
                        label="LLM Provider",
                        choices=list(PROVIDER_PRESETS.keys()),
                        value=llm_provider,
                    )
                with gr.Row():
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
                        value=3.5,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=50,
                    )
            with gr.Column():
                result = gr.Gallery(label="Results", show_label=True)

        MAX_PIXELS = 2**21
        enhance.click(convert_prompt, inputs=[prompt, key, provider], outputs=[prompt])
        width.change(update_max_height, inputs=[width], outputs=[height])
        height.change(update_max_width, inputs=[height], outputs=[width])

        with gr.Column():
            gr.Markdown("### Examples (Enhance prompt finish)")
            for i, ex in enumerate(examples):
                with gr.Row():
                    ex_btn = gr.Button(value=ex, variant="secondary", elem_id=f"ex_btn_{i}", scale=3)
                    ex_img = gr.Image(
                        value=f"img/img_{i + 1}.png",
                        label="Effect",
                        interactive=False,
                        height=130,
                        width=130,
                        scale=1,
                    )
                    ex_btn.click(fn=lambda ex=ex: ex, inputs=[], outputs=prompt)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, num_images],
        outputs=[result, seed],
    )

demo.queue().launch(inbrowser=True)
