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


client = OpenAI(base_url="http://0.0.0.0:9000/v1",api_key="EMPTY")

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:  # open image with binary mode
        image_data = image_file.read()  # read image data
        base64_encoded_data = base64.b64encode(image_data)  # Base64 encode
        base64_string = base64_encoded_data.decode('utf-8')  # convert to utf-8 string
    return base64_string

def parser_object_detection(result_str,image_path):
    import re
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, result_str)
    json_content = match.group(1).strip()

    import json
    result = json.loads(json_content)
    print(colored(result, "green"))

    import cv2

    image_result = cv2.imread(image_path)
    image_result = cv2.resize(image_result, (420,280))
    for item in result:
        bbox_2d = item["bbox_2d"]
        label = item["label"]
        cv2.rectangle(image_result, (bbox_2d[0], bbox_2d[1]), (bbox_2d[2], bbox_2d[3]), (0, 255, 0), 2)
    cv2.imwrite("result.jpg", image_result)

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
    result_str = ""
    for chunk in stream:
        print(colored(chunk.choices[0].delta.content or "", "green"), end="")
        result_str += chunk.choices[0].delta.content or ""
    print("\n")

    parser_object_detection(result_str,image_path)

if __name__ == "__main__":
    prompt="Detect all objects in the image and return their locations in the form of coordinates. The format of output should be like {'bbox': [x1, y1, x2, y2], 'label': the name of this object in English}"
    chat_request(prompt=prompt)


