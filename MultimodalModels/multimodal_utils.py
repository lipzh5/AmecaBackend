# -*- coding:utf-8 -*-
import torch
from collections import deque
from transformers import AutoTokenizer, RobertaTokenizer, RobertaModel, AutoImageProcessor
import torchvision.transforms as transforms
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import time
from PIL import Image
import io
from Utils.FrameBuffer import frame_buffer
from Utils.DialogueBuffer import diag_buffer
# from data_buffers import frame_buffer, diag_buffer
from Const import *
from CONF import *

# face_detector = MTCNN()
from facenet_pytorch import MTCNN
face_detector = MTCNN(keep_all=True, post_process=False, select_largest=False)
'''ref to: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py'''

# ==================↓↓↓ OpenAI API ↓↓↓==================
from openai import AsyncOpenAI
import os.path as osp
file_path = osp.dirname(osp.abspath(__file__))

api_key = ''
try:
	with open(osp.join(file_path, "../openai_key.txt")) as fd:
		api_key = fd.read().strip()
		print('open ai key: ', api_key)
except FileNotFoundError:
	print('could not find openai key file')
    
client = AsyncOpenAI(api_key=api_key)
# ==================↑↑↑ OpenAI API ↑↑↑==================




class Transform:
	def __init__(self):  # cfg: config.data.transform
		self.transform = transforms.Compose(
			[transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  

	def __call__(self, data):
		return self.transform(data)


class Resize:
	def __init__(self, target_size):
		self.target_size = target_size  # cfg.resize.target_size

	def __call__(self, img: np.ndarray):
		interp = cv2.INTER_AREA if img.shape[0] > self.target_size else cv2.INTER_CUBIC
		return cv2.resize(img, dsize=(self.target_size, self.target_size), interpolation=interp)

class Normalize:
	def __init__(self):
		self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	def __call__(self, data):
		return self.normalize(data)

transform = Transform()
resize = Resize(target_size)
normalize = Normalize()


# tokenizer = None
# CONTEXT_CONF = {}
# def get_tokenizer(pretrained_path):
# 	global tokenizer
# 	if tokenizer is None:
# 		tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=False)
# 		_special_tokens_ids = tokenizer('<mask>')['input_ids']
# 		CLS = _special_tokens_ids[0]
# 		MASK = _special_tokens_ids[1]
# 		SEP = _special_tokens_ids[2]
# 		CONTEXT_CONF['CLS'] = CLS  
# 		CONTEXT_CONF['SEP'] = SEP
# 		CONTEXT_CONF['mask_value'] = MASK
# 	return tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=False)
_special_tokens_ids = tokenizer('<mask>')['input_ids']
CLS = _special_tokens_ids[0]
MASK = _special_tokens_ids[1]
SEP = _special_tokens_ids[2]

context_max_len = 256  
context_pad_value = 1  
# CONTEXT_CONF['CLS'] = CLS  
# CONTEXT_CONF['SEP'] = SEP
# CONTEXT_CONF['mask_value'] = MASK


def pad_to_len(sequence_data, max_len, pad_value):
	sequence_data = sequence_data[-max_len:]
	effective_len = len(sequence_data)
	mask = torch.zeros((max_len,))
	mask[:effective_len] = 1

	len_to_pad = max_len - effective_len
	
	if isinstance(sequence_data, list):
		pads = [pad_value]*len_to_pad
		sequence_data.extend(pads)
	elif isinstance(sequence_data, torch.Tensor):
		pads = torch.ones([len_to_pad, *sequence_data.shape[1:]]) * pad_value
		sequence_data = torch.concat((sequence_data, pads))
   
	return sequence_data, mask



def set_vision_encoder(cfg):
	if cfg.model.vision_encoder.model_name == 'inceptionresnetv1':
		use_webface_pretrain = cfg.model.vision_encoder.use_webface_pretrain
		print(f'Inception uses webface pretrain: {use_webface_pretrain} \n ******')
		vision_encoder = InceptionResnetV1(pretrained='casia-webface') if use_webface_pretrain else InceptionResnetV1()
	else:
		# from keras.applications import ResNet50
		use_imgnet_pretrain = cfg.model.vision_encoder.use_imgnet_pretrain
		# vision_encoder = ResNet50(include_top=False, weights='imagenet') if use_imgnet_pretrain else ResNet50(include_top=False) 
		from torchvision.models import resnet50, ResNet50_Weights
		vision_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if use_imgnet_pretrain else resnet50()
		print(f'Resnet50 uses imgnet pretrain: {use_imgnet_pretrain} \n ******')

	vis_enc_trainable = cfg.train.resnet_trainable
	vision_encoder.train(vis_enc_trainable)
	vision_encoder.requires_grad_(vis_enc_trainable)
	return vision_encoder



# ===========================↓↓↓ context modeling ↓↓↓==========================
def get_text_inputs_from_raw():
	query = 'For utterance:'
	query_ids = tokenizer(query)['input_ids'][1:-1]

	utterance_ids = []
	for idx, utt in enumerate(diag_buffer.dialogue):  # TODO optimize we only need the latest one 
		token_ids = tokenizer(utt.decode(encoding))['input_ids'][1:]
		utterance_ids.append(token_ids)
		full_context = [CLS]
		lidx = 0
		for lidx in range(idx):
			total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
			if total_len + len(utterance_ids[idx]) <= context_max_len: # CONFIG['max_len']:
				break
		lidx = max(lidx, idx-8)
		for item in utterance_ids[lidx:]:
			full_context.extend(item)

		query_idx = idx
		# prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
		prompt = 'speaker feels <mask>'
		full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
		input_ids = full_context + full_query
		# print(f'len input ids: {len(input_ids)} \n &&&&&&&&&&&&&&&&&')
		input_ids, _ = pad_to_len(input_ids, max_len=context_max_len, pad_value=context_pad_value) # CONFIG['max_len'], CONFIG['pad_value']
	return torch.tensor(input_ids)
# ===========================↑↑↑ context modeling ↑↑↑==========================

def get_center_faces(img_arr, save_path=None):
	"""extract faces from raw image"""
	boxes, probs = face_detector.detect(img_arr)    # boxes: Nx4 array
	if boxes is None:
		return None
	box_order = np.argsort(np.abs((boxes[:, 2] + boxes[:, 0]) /2 - ORIGINAL_IMG_SHAPE[1]//2))  # [::-1]
	selected_boxes = boxes[0].reshape(-1, 4)
	faces = face_detector.extract(img_arr, selected_boxes, save_path=save_path)
	return faces



def get_vision_inputs_from_raw(n_frames):
	'''ref to: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py'''
	# n_frames = min(int(duration * FPS), MAX_FRAMES)   
	all_faces = []
	ref_face = None
	for i in range(n_frames, 0, -1):
		img_arr = np.asarray(Image.open(io.BytesIO(frame_buffer.buffer_content[-i])))
		'''ablation 1: talking speaker face extraction (together with sound tracking)'''
		# face_tensors = face_detector(img_arr)   # (n, 3, 160, 160)
		# print(f'face tensors: {face_tensors.shape} \n*****')
		face_tensors = get_center_faces(img_arr) 
		if face_tensors is not None:
			n_faces = face_tensors.shape[0]
			for i in range(n_faces):
				face = face_tensors[i]
				'''person-specific normalization'''
				if ref_face is None:
					ref_face = face
				face = face - ref_face  
				face = normalize(face)   # TODO apply normalization???
				all_faces.append(face)
	
	if all_faces:	
		all_faces = torch.stack(all_faces)
		all_faces, mask = pad_to_len(all_faces, MAX_FACES , pad_value=0)
		return all_faces, mask
	mask = torch.zeros([MAX_FACES,])
	mask[:2] = 1    # in case no real faces
	return torch.zeros([MAX_FACES, 3, 160, 160]), mask.long()
	# return np.concatenate(all_faces)  if all_faces else None
