# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: publisher subscriber test (mimic real-time data processing)

import asyncio
import datetime
import zmq
import zmq.asyncio

from Utils import ImagePreprocess
from Const import *
import CONF

context = zmq.asyncio.Context()

img_bytes = ImagePreprocess.convert_img_to_stream()


async def run_server(host, port):
	socket = context.socket(zmq.PUB)
	socket.bind(f'tcp://{host}:{port}')  # use "bind" for publisher and "connect" for subscriber
	print('publisher initialized!!!')
	encoding = CONF.encoding
	while True:
		await asyncio.sleep(1.)
		print(f'send img len in bytes: {len(img_bytes)}')
		query = 'what is this in my hand'.encode(encoding)
		task_type = VisualTasks.VQA.encode(encoding)
		data = [task_type, img_bytes, query]
		if CONF.debug:
			data.append(str(datetime.datetime.now()).encode(encoding))
		await socket.send_multipart(data)
		# await socket.send_multipart([str(datetime.datetime.now()).encode()])

if __name__ == "__main__":
	# import os
	# print(os.getcwd())
	asyncio.run(run_server('127.0.0.1', 2000))


