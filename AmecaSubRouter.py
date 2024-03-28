# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
# import io
from io import BytesIO

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
from PIL import Image

# note: video capture needs configure video_capture node as follows
'''
{
  "data_address": "tcp://0.0.0.0:5001",
  "mjpeg_address": "tcp://0.0.0.0:5000",
  "sensor_name": "Left Eye Camera",
  "video_device": "/dev/eyeball_camera_left",
  "video_height": 720,
  "video_width": 1280
}
'''

ip = '10.6.32.201'
face_detect_addr = f'tcp://{ip}:6669'   # face detection result from Ameca
vsub_addr = f'tcp://{ip}:5000'  # From Ameca, 5000: mjpeg
# vsub_addr = 'tcp://10.126.110.67:5555'  # video capture data subscription
# vsub_sync_addr = 'tcp://10.126.110.67:5555'  # video capture data subscription
vtask_deal_addr = f'tcp://{ip}:2009' #'tcp://10.126.110.67:2006'
# vsub_mjpeg_addr = f'tcp://{ip}:5000'  # mjpeg From Ameca


ctx = Context.instance()


async def on_vqa_task(*args):
	frame = frame_buffer.consume_one_frame()  # TODO merge to blip_analyzer
	if not frame:
		return None
	# res = await run_vqa_from_client_query(frame, *args)
	# return res
	return blip_analyzer.on_vqa_task(frame, *args)


async def on_video_reg_task(*args):
	return video_recognizer.on_video_recognition_task(frame_buffer)


async def on_pose_gen_task(*args):
	return video_recognizer.on_video_rec_posegen_task(frame_buffer)
	# human_action = video_recognizer.on_video_recognition_task(frame_buffer)
	# if human_action is None:
	# 	return None

	# poses = await on_pose_generation(human_action)
	# return [human_action, *poses]


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

		self.face_detect_sub_sock = ctx.socket(zmq.SUB)
		self.face_detect_sub_sock.setsockopt(zmq.SUBSCRIBE, b'')
		# self.face_detect_sub_sock.setsockopt(zmq.CONFLATE, 1)  # do not use this flag, which will cause data loss

		try:
			self.sub_sock.connect(vsub_addr)
		except Exception as e:
			print('Check the ip of Ameca first!!!!')
			print('============================')
			print(str(e))
		try:
			self.face_detect_sub_sock.connect(face_detect_addr)
		except Exception as e:
			print(str(e))
			print('===='*3)
		# self.sub_sock.bind(vsub_addr)

		# context = zmq.Context.instance()
		# self.sub_sock_sync = context.socket(zmq.SUB)
		# self.sub_sock_sync.bind(vsub_sync_addr)
		# self.sub_sock_sync.setsockopt(zmq.SUBSCRIBE, b'')

		self.router_sock = ctx.socket(zmq.ROUTER)
		self.router_sock.connect(vtask_deal_addr)
		# self.router_sock.bind(vtask_deal_addr)

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

	async def sub_face_detect_data(self):
		saved_faces = 0
		last_save_time = time.time()
		try:
			while True:
				data = await self.face_detect_sub_sock.recv_multipart()

				print('face detect data recvd: ', data)  # [xmin, ymin, width, height]
				print([float(t.decode(CONF.encoding)) for t in data])
				print(f'time.time: ', time.time())
				print('---'*5)
				if frame_buffer.buffer_content and saved_faces < 10 and 1 < time.time() - last_save_time: # TODO
					last_save_time = time.time()
					saved_faces += 1
					await self.debug_save_frame(frame_buffer.buffer_content[-1])

				'''
				face detect data recvd:  [b'0.025750000029802322', b'-0.024769999086856842', b'0.1606999933719635', b'0.2856999933719635', b'1711498877.46826']
				[0.025750000029802322, -0.024769999086856842, 0.1606999933719635, 0.2856999933719635, 1711498877.46826]
				time.time:  1711498877.444915
				'''
				# raw_data = await self.face_detect_sub_sock.recv()
				# print('raw data ', raw_data)
			pass
		except Exception as e:
			print(str(e))

	async def debug_save_frame(self, frame_data: bytes):
		img = Image.open(BytesIO(frame_data))
		# img = Image.frombytes('RGB', IMG_SIZE, frame_data)
		img.save(f'{time.time()}.png')
		pass

	async def sub_vcap_data(self):
		try:
			ts = time.time()
			cnt = 0
			while True:
				data = await self.sub_sock.recv()
				frame_buffer.append_content(data)  # TODO time.time() , img = Image.open(BytesIO(data))
				# if cnt < 1:
				# 	img = Image.frombytes('RGB', (1280, 720), data)
				# 	img.save('jpeg_img_from_bytes.png')
				# cnt += 1
				# if cnt == 60:
				# 	total = time.time() - ts
				# 	print(f'total time for receiving 60 frames is {total}')

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
				ans = await self.deal_visual_task(*msg[1:])
				if ans is None:
					ans = 'None'
				print(f'task answer:{ans} \n ------- ')
				resp = [identity, ]
				if isinstance(ans, list) or isinstance(ans, tuple):
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
			# print('deal visual task ans: ', ans)
			return ans
		except Exception as e:
			print(str(e))
			return None
		# return TASK_DISPATCHER[task_type](frame, *tuple(map(lambda x: x.decode(AmecaCONF.encoding), args[1:])))
		# return blip_analyzer.on_vqa_task(frame, args[1].decode(), debug=AmecaCONF.debug)


async def run_sub_router():
	sub_router = SubRouter()
	loop = asyncio.get_event_loop()
	# task = loop.create_task(sub_router.sub_face_detect_data())
	# await asyncio.gather(task)
	task1 = loop.create_task(sub_router.sub_vcap_data())
	task2 = loop.create_task(sub_router.route_visual_task())
	task3 = loop.create_task(sub_router.sub_face_detect_data())
	await asyncio.gather(task1, task2, task3)


if __name__ == "__main__":
	asyncio.run(run_sub_router())




