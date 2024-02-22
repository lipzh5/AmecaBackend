# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import base64
import os


def convert_img_to_stream(img_path=os.path.join(os.getcwd(), '../Assets/img/image_phone.png')):
	with open(img_path, 'rb') as img_file:
		encoded = base64.b64encode(img_file.read())
		# print(type(encoded), len(encoded))
	return encoded


if __name__ == "__main__":
	impath = os.path.join(os.getcwd(), '../Assets/img/image_phone.png')
	encoded = convert_img_to_stream(impath)  # encode
	from PIL import Image
	import io
	import base64
	base64_decoded = base64.b64decode(encoded) # decode
	image = Image.open(io.BytesIO(base64_decoded))
	image.save('saved.png')
