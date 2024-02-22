# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import torch.cuda
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import numpy as np
import base64
import io
import torch

import CONF

MODEL_ID = 'Salesforce/blip-vqa-capfilt-large'


class BlipImageAnalyzer:
	def __init__(self):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to(self.device)
		self.processor = BlipProcessor.from_pretrained(MODEL_ID)
		print(f'Blip model loaded to {self.device}')

	async def generate(self, raw_image: Image, question: str) -> str:
		inputs = self.processor(raw_image, question, return_tensors='pt')
		out = self.model.generate(**inputs, max_new_tokens=100, min_new_tokens=20)
		return self.processor.decode(out[0], skip_special_tokens=True)

	async def on_vqa_task(self, b64_encoded_img, query: bytes):
		decoded = base64.b64decode(b64_encoded_img)  # msg[0] is base64 encoded
		raw_image = Image.open(io.BytesIO(decoded)).convert('RGB')
		await self.generate(raw_image, query.decode(encoding=CONF.encoding))


blip_analyzer = BlipImageAnalyzer()



# def on_vqa_task(b64_encoded_img, bytes_query):
# 	decoded = base64.b64decode(b64_encoded_img)  # msg[0] is base64 encoded
# 	raw_image = Image.open(io.BytesIO(decoded)).convert('RGB')
# 	return run_vqa_from_client_query(raw_image, bytes_query.decode('utf-8'))
#
#
# def run_vqa_from_client_query(raw_image, question):
# 	processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-capfilt-large')
# 	model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-capfilt-large')
# 	# raw_image = Image.open(image_path).convert('RGB')
# 	inputs = processor(raw_image, question, return_tensors='pt')
# 	# print(f'type of inputs: {type(inputs)}') # <class 'transformers.image_processing_utils.BatchFeature'>
# 	out = model.generate(**inputs)
# 	decoded_out = processor.decode(out[0], skip_special_tokens=True)
# 	# print('decoded output: ', decoded_out)
# 	return decoded_out


if __name__ == "__main__":
	raw_img = Image.open('../Assets/image_phone.png').convert('RGB')
	print(f'type of raw img: {type(raw_img)}')
	run_vqa_from_client_query(raw_img, 'what is this in my hand?')
