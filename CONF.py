# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

debug = True  # whether in debug mode or not
encoding = 'utf-8'
frame_buffer_max_len = 640


class HieraConf:
	hiera_temp_stride = 4
	n_infer_clips = 4        # number of clips for inference
	n_frames_per_clip = 16   # TODO 16 , no sampling with stride 4 first
	n_frames_per_video = 64  # 16 * 4
	input_size = (16, 224, 224)
	input_interp_mode = 'trilinear'


gpt_model_name = 'gpt-3.5-turbo-instruct'

# action generation
base_prompt = """
cadidate_poses = [
  'EXP_neutral', 'eyelids_close', 'eyelids_open', 'eyelids_squeeze', 'eyelids_top_down', 'eyelids_wide',
  'HB3_Face_Neutral', 'Lip Corner Smile', 'Mouth Anger', 'Mouth Disgust', 'Mouth Fear', 'Mouth Happy',
  'Mouth Huh', 'Mouth Joy', 'Mouth Open', 'Mouth Sad', 'Mouth Smile', 'Mouth Sneer', 'Mouth Surprise',
  'Mouth Worried','AU4 Jaw Speaking', 'Mouth Neutral', 'AU1 Nose Neutral', 'AU1 Nose Wrinkler'
]
# when the user is reading book
robot.apply_poses(['Mouth Smile', 'eyelids_open']).

# when the user is whistling
robot.apply_poses(['Mouth Surprise', 'eyelids_open']).

# when the user is waiting in line
robot.apply_poses(['Mouth Huh', 'eyelids_close']).
"""





