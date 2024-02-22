# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: action generation using GPT
import os
from openai import OpenAI

file_path = os.path.dirname(os.path.abspath(__file__))
openai_api_key = ''
try:
	with open(os.path.join(file_path, "openai_key.txt")) as fd:
		openai_api_key = fd.read().strip()
		print('open ai key: ', openai_api_key)
except FileNotFoundError:
	print('could not find open ai key file')

client = OpenAI(api_key=openai_api_key)

LLM_CACHE = {}


def gpt_call(engine, prompt, max_token=256, temperature=0, logprobs=1):
	_id = tuple((engine, prompt, max_token, temperature, logprobs))
	resp = LLM_CACHE.get(_id)
	if resp is None:
		resp = client.completions.create(model=engine,
										 prompt=prompt,
										 temperature=temperature,
										 logprobs=logprobs)
		LLM_CACHE[_id] = resp
	return resp




if __name__ == "__main__":
	pass