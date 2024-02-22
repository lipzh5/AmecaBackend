# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import io

import zmq
import zmq.asyncio
from Const import *
import CONF
from VisualModels.BLIP import blip_analyzer


context = zmq.asyncio.Context()

visual_tasks = set()
resp_sending_tasks = set()


async def on_vqa_task(*args):
	if CONF.debug:
		ts = args[-1]
		print(f'on vqa task timestamp: {ts}')
	await blip_analyzer.on_vqa_task(*args[:2])


async def on_video_rec_task(*args):
	pass


async def on_face_rec_task(*args):
	pass

TASK_DISPATCHER = {
	VisualTasks.VQA: on_vqa_task,
	VisualTasks.VideoRecognition: on_video_rec_task,
	VisualTasks.FaceRecognition: on_face_rec_task,
}



def run_background_visual_task(msg, full_cb):

	async def run_task(msg, callback):
		encoding = CONF.encoding
		task_type = msg[0].decode(encoding)
		response = await TASK_DISPATCHER[task_type](*msg[1:])
		# print(f'====msg recvd!!! {type(msg[0])}==== ')
		params = (response.encode(encoding), msg[-1]) if CONF.debug else (response, )
		callback(*params)

	loop = asyncio.get_event_loop()

	coroutine = loop.create_task(
		run_task(msg, full_cb)
	)
	visual_tasks.add(coroutine)
	coroutine.add_done_callback(lambda _:visual_tasks.remove(coroutine))
	# print(f'len of visual tasks: {len(visual_tasks)}')


# ==============================
# TODO should organize these sockets in a tidy way
resp_socket = context.socket(zmq.PUB)
resp_socket.bind('tcp://127.0.0.1:20001')
def on_task_finish_cb(response, timestamp=b''):  # response has already been encoded
	"""should send response to Robot, when visual task is finished """
	async def send_resp_to_robot(resp, ts):
		await resp_socket.send_multipart([resp, ts])
		pass
	loop = asyncio.get_event_loop()

	coroutine = loop.create_task(send_resp_to_robot(response, timestamp))
	resp_sending_tasks.add(coroutine)
	coroutine.add_done_callback(lambda _:resp_sending_tasks.remove(coroutine))
	# print(f'msg has been processed!!! {response}')

async def run_sub():
	socket = context.socket(zmq.SUB)
	# we can connect to several endpoints if we desire, and receive from all
	socket.connect('tcp://127.0.0.1:2000')
	socket.setsockopt(zmq.SUBSCRIBE, b'')
	while True:
		msg = await socket.recv_multipart()
		run_background_visual_task(msg, on_task_finish_cb)


if __name__ == "__main__":
	# arg_test(3, 'name')
	# pass
	asyncio.run(run_sub())
