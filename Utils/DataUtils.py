# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import csv
from Const import *

k400_label_path = os.path.join(os.path.dirname(__file__), '../Assets/data/kinetics_400_labels.csv')


def get_gt_labels_for_k400():
	if not os.path.exists(k400_label_path):
		return {}
	id_to_name_map = {}
	# name_to_id_map = {}
	with open(k400_label_path, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for row in csv_reader:
			_id, name = int(row['id']), row['name']
			id_to_name_map[_id] = name
			# name_to_id_map[name] = _id
	return id_to_name_map #, name_to_id_map


# map video labels to semantic labels (facial expressions)
# semantic or emotion?? TODO
# a list of 400 emotion labels
emotion_labels = [0 for _ in range(400)]
# TODO: annotate manually
# should be map to emotion category using some learned function
# (e.g., a Neural Net, together with **context** or language supervision)
# should be in-context emotion generation
emotion_labels[4] = Emotions.Happy  # applying cream
emotion_labels[57] = Emotions.Surprised  # clapping
emotion_labels[79] = Emotions.Sad  # crying
emotion_labels[80] = Emotions.Fear  # curling hair
emotion_labels[96] = Emotions.Dislike  # doing laundry
emotion_labels[100] = Emotions.Happy   # drinking
emotion_labels[119] = Emotions.Angry   # exercising arm
emotion_labels[127] = Emotions.Surprised  # finger snapping
emotion_labels[131] = Emotions.Dislike   # folding clothes
emotion_labels[132] = Emotions.Scared   # folding napkins
emotion_labels[204] = Emotions.Surprised  # opening present
emotion_labels[210] = Emotions.Dislike    # peeling apples
emotion_labels[218] = Emotions.Happy    # playing badminton
emotion_labels[396] = Emotions.Confused  # writing




