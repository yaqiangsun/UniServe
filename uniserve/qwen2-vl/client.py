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

MODELS_LIST = {
    "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
}

DEFAULT_MODEL = MODELS_LIST["Qwen2-VL-7B"]


client = OpenAI(base_url="http://0.0.0.0:8085/v1",api_key="EMPTY")

def chat_request(prompt="图片中有什么？",image_url = f"file://XXXXXXXXXX.jpg"):
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
   
    
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


