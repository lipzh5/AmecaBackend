# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

N_PIXEL = 1280 * 720 * 3    # num of pixels per frame (Ameca)
IMG_SIZE = (1280, 720)


# visual task types
class VisualTasks:
	VQA = 'VQA'     # TODO str or int?
	VideoRecognition = 'VideoRec'
	VideoRecogPoseGen = 'VideoRecPoseGen'  # video recognition with pose generation
	FaceRecognition = 'FaceRec'
