# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

N_PIXEL = 1280 * 720 * 3    # num of pixels per frame (Ameca)
IMG_SIZE = (1280, 720)


# visual task types
class VisualTasks:
	# PureData = 'PureData'  # pure vcap data
	VQA = 'VQA'     # TODO str or int?
	VideoRecognition = 'VideoRec'
	VideoRecogPoseGen = 'VideoRecPoseGen'  # video recognition with pose generation
	FaceRecognition = 'FaceRec'


def func(*args):
	print('func args, ', args)
	pass

if __name__ == "__main__":
	t = [1,2]
	s = [3, *t]
	print(s)
	# t = (b'hello',)
	# print(map(lambda x: x.decode(), t))
	# func('user func', *tuple(map(lambda x: x.decode(), t)))
