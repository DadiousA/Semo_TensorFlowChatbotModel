# -*- coding:utf-8 -*-
import numpy as np
import time
import sys
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from dynamic_seq2seq import dynamicSeq2seq
import jieba

import vectorize


class seq2seq():
	'''
	tensorflow-1.0.0

		args:
		encoder_vec_file    encoder向量文件
		decoder_vec_file    decoder向量文件
		encoder_vocabulary  encoder词典
		decoder_vocabulary  decoder词典
		model_path          模型目录
		batch_size          批处理数
		sample_num          总样本数
		max_batches         最大迭代次数
		show_epoch          保存模型步长

	'''

	def __init__(self, tf):
		self.tf = tf
		print("tensorflow version: ", tf.__version__)
		self.tf.reset_default_graph()
		self.encoder_vec_file = vectorize.ENCODER_VEC
		self.decoder_vec_file = vectorize.DECODER_VEC
		self.encoder_vocabulary = vectorize.ENCODER_VOCAB
		self.decoder_vocabulary = vectorize.DECODER_VOCAB
		self.dictFile = './materials/word_dict.txt'
		self.model_path = './model/'
		self.batch_size = 40
		self.max_batches = 4000
		self.show_epoch = 500
		self.sample_num = 0

		jieba.load_userdict(self.dictFile)

		self.model = dynamicSeq2seq(encoder_cell=LSTMCell(40),
									decoder_cell=LSTMCell(40),
									encoder_vocab_size=vectorize.MAX_VOCAB_NUM,
									decoder_vocab_size=vectorize.MAX_VOCAB_NUM,
									embedding_size=40,
									attention=True,
									bidirectional=False,
									debug=False,
									time_major=True)
		self.location = ["杭州", "重庆", "上海", "北京"]
		# self.user_info = {"__username__": "yw", "__location__": "重庆"}
		# self.robot_info = {"__robotname__": "Rr"}
		self.dec_vocab = {}
		self.enc_vocab = {}
		self.dec_vecToSeg = {}
		tag_location = ''
		with open(self.encoder_vocabulary, "r") as enc_vocab_file:
			for index, word in enumerate(enc_vocab_file.readlines()):
				self.enc_vocab[word.strip()] = index
		with open(self.decoder_vocabulary, "r") as dec_vocab_file:
			for index, word in enumerate(dec_vocab_file.readlines()):
				self.dec_vecToSeg[index] = word.strip()
				self.dec_vocab[word.strip()] = index
		print("Model created")




	def get_feed_dict(self, train_inputs, train_targets, batches, sample_num):
		'''获取batch

			为向量填充PAD
			最大长度为每个batch中句子的最大长度
			并将数据作转换:
			[batch_size, time_steps] -> [time_steps, batch_size]

		'''
		batch_inputs = []
		batch_targets = []
		batch_inputs_length = []
		batch_targets_length = []

		pad_inputs = []
		pad_targets = []

		# 随机样本
		shuffle = np.random.randint(0, sample_num, batches)

		#print(shuffle)

		en_max_seq_length = max([len(train_inputs[i]) for i in shuffle])
		de_max_seq_length = max([len(train_targets[i]) for i in shuffle])

		for index in shuffle:
			# get the vector of sample and store in _en
			_en = train_inputs[index]
			inputs_batch_major = np.zeros(shape=[en_max_seq_length], dtype=np.int32)  # == PAD
			for seq in range(len(_en)):
				inputs_batch_major[seq] = _en[seq]
			# add PAD and store vector in batch_input
			batch_inputs.append(inputs_batch_major)
			# add the batch size
			batch_inputs_length.append(len(_en))

			_de = train_targets[index]
			inputs_batch_major = np.zeros(shape=[de_max_seq_length], dtype=np.int32)  # == PAD
			for seq in range(len(_de)):
				inputs_batch_major[seq] = _de[seq]
			batch_targets.append(inputs_batch_major)
			batch_targets_length.append(len(_de))

		#print(sum(shuffle))

		batch_inputs = np.array(batch_inputs).swapaxes(0, 1)
		batch_targets = np.array(batch_targets).swapaxes(0, 1)

		#### batch input, batch_input_length
		return {self.model.encoder_inputs: batch_inputs,
				self.model.encoder_inputs_length: batch_inputs_length,
				self.model.decoder_targets: batch_targets,
				self.model.decoder_targets_length: batch_targets_length, }

	def train(self):

		# 获取输入输出,存储为vector arrays
		def data_set(vecfile):
			_ids = []
			counter = 0
			with open(vecfile, "r") as fw:
				
				for line in fw.readlines():
					counter += 1
					line = [int(i) for i in line.split()]
					_ids.append(line)
				print(counter)
			return _ids
		train_inputs = data_set(self.encoder_vec_file)
		train_targets = data_set(self.decoder_vec_file)
		self.sample_num = len(train_inputs)
		print("共有 %s 条样本" % self.sample_num)

		# save configurations
		config = self.tf.ConfigProto()
		config.gpu_options.allow_growth = True

		# start training
		with self.tf.Session(config=config) as sess:

			# 初始化变量
			ckpt = self.tf.train.get_checkpoint_state(self.model_path)
			if ckpt is not None:
				print("Model Path", ckpt.model_checkpoint_path)
				self.model.saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				sess.run(tf.global_variables_initializer())

			loss_track = []
			total_time = 0
			for batch in range(self.max_batches + 1):
				# 获取feed_dict [time_steps, batch_size]
				start = time.time()
				feed_dict = self.get_feed_dict(train_inputs,
								 train_targets,
								 self.batch_size,
								 self.sample_num)
				_, loss, _, _ = sess.run([self.model.train_op,
										  self.model.loss,
										  self.model.gradient_norms,
										  self.model.updates], feed_dict)

				#loss= sess.run(self.model.loss, feed_dict)

				# print("wtf",loss)



				stop = time.time()
				total_time += (stop - start)

				# if batch > 0 and loss > loss_track[-1]:
				#     sess.run(self.model.learning_rate_decay_op)
				#     continue

				loss_track.append(loss)

				if batch == 0 or batch % self.show_epoch == 0:

					print("-" * 50)
					print("n_epoch {}".format(sess.run(self.model.global_step)))
					print('  minibatch loss: ', loss)
					print('  per-time: %s' % (total_time / self.show_epoch))
					checkpoint_path = self.model_path + "chatbot_seq2seq.ckpt"
					# 保存模型
					

					self.model.saver.save(sess, checkpoint_path, global_step=self.model.global_step)

					# 清理模型
					self.clearModel()

	def addToFile(self, strs, file):
		with open(file, "a") as f:
			f.write(strs + "\n")

	def addVocab(self, word, kind):
		if kind == 'enc':
			self.addToFile(word, self.encoder_vocabulary)
			index = max(self.enc_vocab.values()) + 1
			self.enc_vocab[word] = index
		else:
			self.addToFile(word, self.decoder_vocabulary)
			index = max(self.dec_vocab.values()) + 1
			self.dec_vocab[word] = index
			self.dec_vecToSeg[index] = word
		return index

 
	def segement(self, strs):
		return jieba.lcut(strs)

	def make_inference_feed_dict(self, inputs_seq):
		sequence_lengths = [len(seq) for seq in inputs_seq]
		max_seq_length = max(sequence_lengths)
		batch_size = len(inputs_seq)

		inputs_time_major = []
		# PAD
		for sents in inputs_seq:
			inputs_batch_major = np.zeros(shape=[max_seq_length], dtype=np.int32)  # == PAD
			for index in range(len(sents)):
				inputs_batch_major[index] = sents[index]
			inputs_time_major.append(inputs_batch_major)

		inputs_time_major = np.array(inputs_time_major).swapaxes(0, 1)
		return {self.model.encoder_inputs: inputs_time_major,
				self.model.encoder_inputs_length: sequence_lengths}

	def predict(self):
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(self.model_path)
			if ckpt is not None:
				print(ckpt.model_checkpoint_path)
				self.model.saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print("没找到模型")

			action = False
			while True:
				if not action:
					inputs_strs = input("me > ")
				if not inputs_strs:
					continue

				inputs_strs = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", inputs_strs)

				action = False
				segements = self.segement(inputs_strs)
				# inputs_vec = [enc_vocab.get(i) for i in segements]
				inputs_vec = []
				for i in segements:
					if i in self.location:
						tag_location = i
						Action.tag_location = i
						inputs_vec.append(self.enc_vocab.get("__location__", self.model.UNK))
						continue
					inputs_vec.append(self.enc_vocab.get(i, self.model.UNK))
				feed_dict = self.make_inference_feed_dict([inputs_vec])
				inf_out = sess.run(self.model.decoder_prediction_inference, feed_dict)
				inf_out = [i[0] for i in inf_out]

				outstrs = ''
				for vec in inf_out:
					if vec == self.model.EOS:
						break
					outstrs += self.dec_vecToSeg.get(vec, self.model.UNK)
				print(outstrs)

	def clearModel(self, remain=3):
		try:
			filelists = os.listdir(self.model_path)
			re_batch = re.compile(r"chatbot_seq2seq.ckpt-(\d+).")
			batch = re.findall(re_batch, ",".join(filelists))
			batch = [int(i) for i in set(batch)]
			if remain == 0:
				for file in filelists:
					if "chatbot_seq2seq" in file:
						os.remove(self.model_path + file)
				os.remove(self.model_path + "checkpoint")
				return
			if len(batch) > remain:
				for bat in sorted(batch)[:-(remain)]:
					for file in filelists:
						if str(bat) in file and "chatbot_seq2seq" in file:
							os.remove(self.model_path + file)
		except Exception as e:
			return


if __name__ == '__main__':


	seq = seq2seq(tf)
	seq.clearModel(0)
	seq.train()
	seq.predict()
