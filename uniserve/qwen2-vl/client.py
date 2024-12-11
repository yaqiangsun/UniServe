# -*- coding: utf-8 -*-
# Copyright (c) 2024 yaqiang.sun.
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
#########################################################################
# Author: yaqiangsun
# Created Time: 2024/12/10 17:25:05
########################################################################


import argparse
from openai import OpenAI
from termcolor import colored
import base64

MODELS_LIST = {
    "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
}


DEFAULT_MODEL = MODELS_LIST["Qwen2-VL-7B"]


client = OpenAI(base_url="http://0.0.0.0:8085/v1",api_key="EMPTY")

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:  # open image with binary mode
        image_data = image_file.read()  # read image data
        base64_encoded_data = base64.b64encode(image_data)  # Base64 encode
        base64_string = base64_encoded_data.decode('utf-8')  # convert to utf-8 string
    return base64_string

def chat_request(prompt="图片中有什么？",image_url = f"file://XXXXXXXXXX.jpg"):

    # image url as image input
    # image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

    # read image file and encode to base64 as input
    image_path = "./demo.jpeg"
    image_data = image_to_base64(image_path)
    image_url = f"data:image/jpeg;base64,{image_data}"
   
    
    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "image_url": image_url,
                        "resized_height": 280,
                        "resized_width": 420
                    },
                ],
            }
        ],
        stream=True,
        max_tokens=1024,
    )
    print(colored("Runing...", "yellow"))
    for chunk in stream:
        print(colored(chunk.choices[0].delta.content or "", "green"), end="")

if __name__ == "__main__":
    chat_request()


