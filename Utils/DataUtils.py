# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import csv

k400_label_path = os.path.join(os.path.dirname(__file__), '../Assets/data/kinetics_400_labels.csv')


def get_gt_labels_for_k400():
	if not os.path.exists(k400_label_path):
		return {}
	id_to_name_map = {}
	with open(k400_label_path, mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for row in csv_reader:
			id_to_name_map[int(row['id'])] = row['name']
	return id_to_name_map

