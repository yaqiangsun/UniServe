# -*- coding: utf-8 -*-
# Copyright (c) 2025 yaqiang.sun.
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
#########################################################################
# Author: yaqiangsun
# Created Time: 2025/07/10 16:19:48
########################################################################


import litserve as ls
from litserve.specs.openai import ChatCompletionRequest
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from PIL import Image
import torch
import io
import base64
from threading import Thread
from qwen_vl_utils import process_vision_info
import os



# Add exception handling to thread
class ThreadWithExceptionHandling(Thread):
    def __init__(self, target, args=(), kwargs=None):
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.target = target
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
    def run(self):
        try:
            self.target(*self.args, **self.kwargs)
        except Exception as e:
            print(f"Exception in thread: {e}")


class QwenVLAPI(ls.LitAPI):
    def setup(self, device):
        # 多模态模型加载
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"Model loaded on {device}")

        self.streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def decode_request(self, request: ChatCompletionRequest , context: dict):
        # reload model
        # if request.model != self.model_id:
        #     self.setup(self.device, request.model)
   
        context["generation_args"] = {
            "max_new_tokens": request.max_tokens if request.max_tokens else 2048,
        }
        messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]

        # add resized_height and resized_width to image
        for message in (messages):
            if message["role"] == "user":
                for content_item in message["content"]:
                    if content_item.get("type") == "image":
                        content_item.update(
                            {
                                "resized_height": 280,
                                "resized_width": 420
                            }
                        )

   
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs

    def predict(self, model_inputs, context: dict):
        generation_kwargs = dict(
            model_inputs,
            streamer=self.streamer,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **context["generation_args"],
        )
        thread = ThreadWithExceptionHandling(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in self.streamer:
            yield text
if __name__ == "__main__":
    import qwen_specs
    api = QwenVLAPI(
                    # spec=ls.OpenAISpec(),
                    spec=qwen_specs.OpenAISpec(),
                    # max_batch_size=4,        # 批处理提升吞吐量 但是需要修改解析形式
                    )
    server = ls.LitServer(api,
                          devices=1,          # 自动检测GPU "auto"
                          workers_per_device=1,    # 每GPU进程数
                          accelerator="cuda",
                          )
    server.run(port=9000, host="0.0.0.0")
