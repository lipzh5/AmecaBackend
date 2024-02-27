# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import aiohttp
import asyncio
import os
import time
import base64
from PIL import Image
import io
import CONF

file_path = os.path.dirname(os.path.abspath(__file__))
api_key = ''
try:
	with open(os.path.join(file_path, "../openai_key.txt")) as fd:
		api_key = fd.read().strip()
		print('open ai key: ', api_key)
except FileNotFoundError:
	print('could not find open ai key file')


# def encode_image(image_path):
# 	with open(image_path, 'rb') as image_file:
# 		return base64.b64encode(image_file.read()).decode('utf-8')


headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"
	}

payload = {
		"model": "gpt-4-vision-preview",
		"messages": [
					{
						"role": "user",
					}
				],
		"max_tokens": 300
}


async def run_vqa_from_client_query(img_bytes, query:bytes):
	if not api_key:
		return
	query = query.decode(CONF.encoding)
	img = Image.frombytes('RGB', (1280, 720), img_bytes)
	b = io.BytesIO()
	img.save(b, 'png')
	base64_image = base64.b64encode(b.getvalue()).decode('utf-8')
	print(f'run vqa after encode image: {time.strftime("%X")}')
	# update query and image
	content = [
		{
			"type": "text",
			"text": query,
		},
		{
			"type": "image_url",
			"image_url": {
				"url": f"data:image/jpeg;base64,{base64_image}"
			}
		}
	]
	payload['messages'][0]['content'] = content
	print(f'run vqa starts send request: {time.strftime("%X")}')
	async with aiohttp.ClientSession() as session:
		response = await session.post(url="https://api.openai.com/v1/chat/completions",
									  headers=headers,
									  json=payload)
		res = await response.json()
		print(res)
		print(res['choices'][0])
		# print('type of message: ', type(res['choices'][0]['message']))
		print(res['choices'][0]['message'])
		print(f'run vqa get answer: {time.strftime("%X")}')
		return res['choices'][0]['message']['content']


if __name__ == "__main__":
	# asyncio.run(video_captioning('../saved_frames_rollout/batch3'))
	query = 'what is this in my hand?'
	img_path = os.path.dirname(os.path.abspath(__file__))
	img_path = os.path.join(img_path, '../assets/image_hair_dryer.png')
	# print(img_path)
	# print(f'os path exists? {os.path.exists(img_path)}')
	# print(time.strftime('%X'))
	asyncio.run(run_vqa_from_client_query(query, None, img_path))
	# print(time.strftime('%X'))

	pass




