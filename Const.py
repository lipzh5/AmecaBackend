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
	EmotionImitation = 'EmoImitation' 

class NLPTask:
	RAG = 'RAG'  # retrieve augmented generation task


class ResponseCode:
	KeepSilent = b'0' 
	Success = b'1'
	Fail = b'2'



class Emotions:
	Other = -1
	Neutral = 0
	Angry = 1
	Confused = 2
	Dislike = 3
	Fear = 4
	Happy = 5
	Sad = 6
	Scared = 7
	Surprised = 8


EMOTION_TO_ANIM = {
	Emotions.Other: ['Chat Expressions.dir/Chat_G2_Neutral.project',],
	Emotions.Neutral: ['Chat Expressions.dir/Chat_G2_Neutral.project',],
	Emotions.Angry: [
		'Chat Expressions.dir/Chat_G2_Angry_1.project',
		'Chat Expressions.dir/Chat_G2_Angry_2.project',
		'Chat Expressions.dir/Chat_G2_Angry_3.project',],
	Emotions.Confused: ['Chat Expressions.dir/Chat_G2_Confused_1.project',],
	Emotions.Dislike: [
		'Chat Expressions.dir/Chat_G2_Dislike_1.project',
		'Chat Expressions.dir/Chat_G2_Dislike_2.project',],
	Emotions.Fear: [
		'Chat Expressions.dir/Chat_G2_Fear_1.project',
		'Chat Expressions.dir/Chat_G2_Fear_2.project',],
	Emotions.Happy: [
		'Chat Expressions.dir/Chat_G2_Happy_1.project',
		'Chat Expressions.dir/Chat_G2_Happy_2.project',
		'Chat Expressions.dir/Chat_G2_Happy_with_audio.project',],
	Emotions.Sad: [
		'Chat Expressions.dir/Chat_G2_Sad_1.project',
		'Chat Expressions.dir/Chat_G2_Sad_2.project',],
	Emotions.Scared: ['Chat Expressions.dir/Chat_G2_Scared_1.project',],
	Emotions.Surprised: [
		'Chat Expressions.dir/Chat_G2_Surprised_1.project',
		'Chat Expressions.dir/Chat_G2_Surprised_2.project',]


}





def func(*args):
	print('func args, ', args)
	pass

if __name__ == "__main__":
	t = (3, 4)
	func(*t)
	# t = (b'hello',)
	# print(map(lambda x: x.decode(), t))
	# func('user func', *tuple(map(lambda x: x.decode(), t)))
