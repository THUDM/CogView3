import re
import argparse
from openai import OpenAI
import traceback


def clean_string(s):
    s = s.replace("\n", " ")
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def upsample_prompt(
        prompt: str,
        api_key: str,
        url: str,
        model: str
) -> str:
    client = OpenAI(api_key=api_key, base_url=url)
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
            model=model,
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=300,
        )
        prompt = response.choices[0].message.content
        if prompt:
            prompt = clean_string(prompt)
    except Exception as e:
        traceback.print_exc()
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, help="api key")
    parser.add_argument("--prompt", type=str, help="Prompt to upsample")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://open.bigmodel.cn/api/paas/v4",
        help="base url"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="glm-4-plus",
        help="LLM using for upsampling"
    )
    args = parser.parse_args()

    api_key = args.api_key
    prompt = args.prompt

    prompt_enhanced = upsample_prompt(
        prompt=prompt,
        api_key=api_key,
        url=args.base_url,
        model=args.model
    )
    print(prompt_enhanced)
