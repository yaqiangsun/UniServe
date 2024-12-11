# -*- coding: utf-8 -*-
# Copyright (c) 2024 yaqiang.sun.
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
#########################################################################
# Author: yaqiangsun
# Created Time: 2024/12/10 17:24:55
########################################################################

import torch
from transformers import (AutoProcessor,Qwen2VLForConditionalGeneration,TextIteratorStreamer)
from transformers import BitsAndBytesConfig

import litserve as ls
from litserve.specs.openai import ChatCompletionRequest

from qwen_vl_utils import process_vision_info

from threading import Thread


MODELS_LIST = {
    "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
}

DEFAULT_MODEL = MODELS_LIST["Qwen2-VL-7B"]

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

class Qwen2VLServe(ls.LitAPI):
    def setup(self, device, model_id:str=DEFAULT_MODEL):
        if model_id not in MODELS_LIST.values():
            return ls.LitError(f"Invalid model ID: {model_id}")
            # raise ValueError(f"Invalid model ID: {model_id}")

        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #_attn_implementation="flash_attention_2",
            device_map="balanced",
            # quantization_config=quantization_config,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.device = device
        self.model_id = model_id
    def decode_request(self, request: ChatCompletionRequest , context: dict):
        # reload model
        if request.model != self.model_id:
            self.setup(self.device, request.model)
   
        context["generation_args"] = {
            "max_new_tokens": request.max_tokens if request.max_tokens else 2048,
        }
        messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]
   
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
    api = Qwen2VLServe()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8085)
