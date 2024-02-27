# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import zmq
import zmq.asyncio
from zmq.asyncio import Context
from Utils import ImagePreprocess
from Utils.FrameBuffer import frame_buffer
from VisualModels.BLIP import blip_analyzer
from VisualModels.GPT4V import run_vqa_from_client_query
from VisualModels.Hiera import video_recognizer
from ActionGeneration import on_pose_generation
from VisualModels.InsightFace import find_from_db
import CONF
from Const import *
import time
import base64

# vsub_addr = 'tcp://10.9.8.164:5001'
vsub_addr = 'tcp://10.126.110.67:5555'  # video capture data subscription
# vsub_sync_addr = 'tcp://10.126.110.67:5555'  # video capture data subscription
vtask_deal_addr = 'tcp://10.126.110.67:2009'

ctx = Context.instance()


async def on_vqa_task(*args):
	frame = frame_buffer.consume_one_frame()  # TODO merge to blip_analyzer
	if not frame:
		return None
	res = await run_vqa_from_client_query(frame, *args)
	return res
	# return blip_analyzer.on_vqa_task(frame, *args)


async def on_video_reg_task(*args):
	return video_recognizer.on_video_recognition_task(frame_buffer)


async def on_pose_gen_task(*args):
	human_action = video_recognizer.on_video_recognition_task(frame_buffer)
	if human_action is None:
		return None
	poses = await on_pose_generation(human_action)
	return [human_action, *poses]


async def on_face_rec_task(*args):
	try:
		for i in range(CONF.face_reg_try_cnt):
			found = find_from_db(frame_buffer.buffer_content[-i-1])
			if found:
				return found
	except Exception as e:
		print(str(e))
		return None


TASK_DISPATCHER = {
	VisualTasks.VQA: on_vqa_task,
	VisualTasks.VideoRecognition: on_video_reg_task,
	VisualTasks.VideoRecogPoseGen: on_pose_gen_task,
	VisualTasks.FaceRecognition: on_face_rec_task,
}


class SubRouter:
	def __init__(self):
		super().__init__()
		self.sub_sock = ctx.socket(zmq.SUB)
		self.sub_sock.setsockopt(zmq.SUBSCRIBE, b'')
		self.sub_sock.setsockopt(zmq.CONFLATE, 1)
		# self.sub_sock.connect(vsub_addr)
		self.sub_sock.bind(vsub_addr)


		# context = zmq.Context.instance()
		# self.sub_sock_sync = context.socket(zmq.SUB)
		# self.sub_sock_sync.bind(vsub_sync_addr)
		# self.sub_sock_sync.setsockopt(zmq.SUBSCRIBE, b'')

		self.router_sock = ctx.socket(zmq.ROUTER)
		self.router_sock.bind(vtask_deal_addr)

	# def sync_sub_vcap_data(self):
	# 	try:
	# 		while True:
	# 			msg = self.sub_sock_sync.recv_multipart()
	# 			frame_buffer.append_content(msg[0])
	# 			content_len = len(frame_buffer.buffer_content)
	# 			# if content_len % 64 == 0:
	# 			# 	print(f'sync len frame buffer content: {len(frame_buffer.buffer_content)}')
	# 	except Exception as e:
	# 		print(str(e))

	# def test_base64_encode(self, data):
	# 	encoded = base64.b64encode(data)
	# 	print('encode data successful ', len(data))
	# 	pass

	async def sub_vcap_data(self):
		try:
			while True:
				data = await self.sub_sock.recv()
				# print(f'len of data: {len(data)}, time: {time.time} ')
				frame_buffer.append_content(data)
				# msg = await self.sub_sock.recv_multipart()
				# frame_buffer.append_content(msg[0])
				# print(f'len frame buffer content: {len(frame_buffer.buffer_content)}')
		except Exception as e:
			print(str(e))

	async def route_visual_task(self):
		try:
			while True:
				msg = await self.router_sock.recv_multipart()
				print('router sock recv msg: ', msg)
				print('------')
				identity = msg[0]
				print('route visual task identity: ', identity)
				# await self.router_sock.send_multipart([identity, b'visual task resp'])
				# ans = await on_vqa_task(msg[1])
				ans = await self.deal_visual_task(*msg[1:])  # TODO test
				if ans is None:
					ans = 'None'
				print(f'task answer:{ans} \n ------- ')
				resp = [identity, ]
				if isinstance(ans, list):
					resp.extend([item.encode(CONF.encoding) for item in ans])
				else:
					resp.append(ans.encode(CONF.encoding))
				await self.router_sock.send_multipart(resp)
		except Exception as e:
			print(str(e))

	async def deal_visual_task(self, *args):
		try:
			task_type = args[0].decode(CONF.encoding)
			print('deal visual task type: ', task_type)
			print('=====')
			ans = await TASK_DISPATCHER[task_type](*args[1:])
			# ans = blip_analyzer.on_vqa_task(frame, args[1], debug=CONF.debug)
			print('deal visual task ans: ', ans)
			return ans
		except Exception as e:
			print(str(e))
			return None
		# return TASK_DISPATCHER[task_type](frame, *tuple(map(lambda x: x.decode(AmecaCONF.encoding), args[1:])))
		# return blip_analyzer.on_vqa_task(frame, args[1].decode(), debug=AmecaCONF.debug)


async def run_sub_router():
	sub_router = SubRouter()
	loop = asyncio.get_event_loop()
	# task = loop.create_task(sub_router.route_visual_task())
	task1 = loop.create_task(sub_router.sub_vcap_data())
	task2 = loop.create_task(sub_router.route_visual_task())
	await asyncio.gather(task1, task2)


if __name__ == "__main__":
	asyncio.run(run_sub_router())




