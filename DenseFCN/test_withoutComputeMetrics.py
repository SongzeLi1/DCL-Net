from __future__ import print_function
import os

# choose one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import warnings
import numpy as np
import tensorflow as tf
from datetime import datetime

slim = tf.contrib.slim
from skimage import io
import utils
from PIL import Image
import cv2
from skimage import transform
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import denseFCN  # Proposed model

import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS
# When testing, the batch size is set to be 1#
tf.flags.DEFINE_integer('batch_size', 1, 'batch size')


# tf.flags.DEFINE_string('data_dir',
#                         '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_imgs/',
#                        'path to dataset')
# tf.flags.DEFINE_string('data_dir',
#                         '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/img/',
#                        'path to dataset')
# tf.flags.DEFINE_string('data_dir',
#                         '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_imgs/',
#                        'path to dataset')
# tf.flags.DEFINE_string('data_dir',
#                         '/data1/zhengkengtao/docimg/tamper_mosaic/',
#                        'path to dataset')
# tf.flags.DEFINE_string('data_dir',
#                         '/data1/zhengkengtao/exps/0801_denseFCN_Alinew_trainsplit811_resize512x512Aug/test_docimgall/left/',
#                        'path to dataset')
# tf.flags.DEFINE_string('restore',
# 					   '/data1/zhengkengtao/exps/0717_denseFCN_Alinew_trainsplit811_crop512x512/model.ckpt-0.214969-0.624936-43',
# 					   'Explicitly restore checkpoint')
# tf.flags.DEFINE_string('data_dir',
#                         '/data1/zhengkengtao/exps/1023_denseFCN_docimg_split811_crop512selectnoblank_OrigMosaicOneKind/continue/test_docimgsplit811test_0.256614-0.589943-54/SizeTooBigTestImgs/',
#                        'path to dataset')
# tf.flags.DEFINE_string('restore',
# 					   '/data1/zhengkengtao/exps/1023_denseFCN_docimg_split811_crop512selectnoblank_OrigMosaicOneKind/continue/model.ckpt-0.256614-0.589943-54',
# 					   'Explicitly restore checkpoint')
# tf.flags.DEFINE_string('data_dir',
#                         '/pubdata/zhengkengtao/1130/SizeTooBigTestImgs/',
#                        'path to dataset')
# tf.flags.DEFINE_string('restore',
# 					   '/pubdata/zhengkengtao/1130/model.ckpt-0.256614-0.589943-54',
# 					   'Explicitly restore checkpoint')
# tf.flags.DEFINE_string('data_dir',
#                         '/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride256/',
#                        'path to dataset')
# tf.flags.DEFINE_string('restore',
# 					   '/data1/zhengkengtao/exps/0609_denseFCN_docimg_split811_crop512x512selectnoblank/model.ckpt-0.80231-0.981443-53',
# 					   'Explicitly restore checkpoint')
# tf.flags.DEFINE_string('data_dir',
#                         '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split811/test_imgs/',
#                        'path to dataset')
# tf.flags.DEFINE_string('restore',
# 					   '/pubdata/zhengkengtao/exps/0801_denseFCN_Alinew_trainsplit811_resize768x768Aug/model.ckpt-0.179086-0.778738-40',
# 					   'Explicitly restore checkpoint')
tf.flags.DEFINE_string('data_dir',
                        '/data1/zhengkengtao/docimg/tamper_mosaic_crop512stride256/',
                       'path to dataset')
tf.flags.DEFINE_string('restore',
					   '/data1/zhengkengtao/exps/0717_denseFCN_Alinew_trainsplit811_crop512x512/model.ckpt-0.214969-0.624936-43',
					   'Explicitly restore checkpoint')


# 2021.11.07
test_imgs_txt = None
# test_imgs_txt = '/pubdata/zhengkengtao/certificate/val[0.1].txt'
test_result_save_path = '/data1/zhengkengtao/exps/0717_denseFCN_Alinew_trainsplit811_crop512x512/testdocimgall_crop512stride256/'
threshold = 0.5

tf.flags.DEFINE_string('visout_dir',
                       test_result_save_path + 'unthreshold/',
                       'path to output unthresholded predict maps')
tf.flags.DEFINE_string('visout_threshold_dir',
                       test_result_save_path + 'threshold_{}/'.format(threshold),
                       'path to output thresholded predict maps (use 0.5)')
tf.flags.DEFINE_string('record_path',
                       test_result_save_path + 'metrics/',
                       'path to output a recording file ')

if (os.path.exists(FLAGS.visout_dir) == False):
	os.makedirs(FLAGS.visout_dir)
if (os.path.exists(FLAGS.visout_threshold_dir) == False):
	os.makedirs(FLAGS.visout_threshold_dir)
if (os.path.exists(FLAGS.record_path) == False):
	os.makedirs(FLAGS.record_path)
f = open(os.path.join(FLAGS.record_path, "log.txt"), 'w+')

'''In testing phase, the following setting is ignored'''
tf.flags.DEFINE_integer('subset', None, 'Use a subset of the whole dataset')
# tf.flags.DEFINE_string('img_size', '3000x3000', 'size of input image') # None docimgsplit811test有4张图像过大原尺寸测不了
tf.flags.DEFINE_string('img_size', None, 'size of input image') # None docimgsplit811test有4张图像过大原尺寸测不了
tf.flags.DEFINE_bool('img_aug', None, 'apply image augmentation')
tf.flags.DEFINE_string('mode', 'test', 'Mode: train / test / visual')
tf.flags.DEFINE_integer('epoch', 30, 'No. of epoch to run')
tf.flags.DEFINE_float('train_ratio', 1.0, 'Trainning ratio')

tf.flags.DEFINE_bool('reset_global_step', True, 'Reset global step')
tf.flags.DEFINE_integer('test_img_num', len(os.listdir(FLAGS.data_dir)), 'Test image num')
# learning configuration
tf.flags.DEFINE_string('optimizer', 'Adam', 'GradientDescent / Adadelta / Momentum / Adam / Ftrl / RMSProp')
tf.flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_float('lr_decay', 0.5, 'Decay of learning rate')
tf.flags.DEFINE_float('lr_decay_freq', 1.0, 'Epochs that the lr is reduced once')
tf.flags.DEFINE_string('loss', 'xent', 'Loss function type')
tf.flags.DEFINE_float('focal_gamma', '2.0', 'gamma of focal loss')
tf.flags.DEFINE_float('weight_decay', 5e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_integer('shuffle_seed', None, 'Seed for shuffling images')
tf.flags.DEFINE_integer('verbose_time', 20, 'verbose times in each epoch')
tf.flags.DEFINE_integer('valid_time', 1, 'validation times in each epoch')
tf.flags.DEFINE_integer('keep_ckpt', 0, 'num of checkpoint files to keep')

print("Batch size:", str(FLAGS.batch_size), " , optimizer: ", FLAGS.optimizer, ", Learning rate: ",
      str(FLAGS.learning_rate),
      ", lr decay: ", str(FLAGS.lr_decay), " , Lr decay freq: ", str(FLAGS.lr_decay_freq), " , loss: " + FLAGS.loss)

OPTIMIZERS = {
	'GradientDescent': {'func': tf.train.GradientDescentOptimizer, 'args': {}},
	'Adadelta': {'func': tf.train.AdadeltaOptimizer, 'args': {}},
	'Momentum': {'func': tf.train.MomentumOptimizer, 'args': {'momentum': 0.9}},
	'Adam': {'func': tf.train.AdamOptimizer, 'args': {}},
	'Ftrl': {'func': tf.train.FtrlOptimizer, 'args': {}},
	'RMSProp': {'func': tf.train.RMSPropOptimizer, 'args': {}}
}
LOSS = {
	'wxent': {'func': utils.losses.sparse_weighted_softmax_cross_entropy_with_logits, 'args': {}},
	'focal': {'func': utils.losses.focal_loss, 'args': {'gamma': FLAGS.focal_gamma}},
	'f1': {'func': utils.losses.quasi_f1_loss, 'args': {}},
	'xent': {'func': utils.losses.sparse_softmax_cross_entropy_with_logits, 'args': {}}
}


def model(images, weight_decay, is_training, num_classes=2):
	# # 原始的denseFCN，用BN层
	return denseFCN.denseFCN(images, is_training,weight_decay,num_classes)

	# 改动的denseFCN，用IN层
	# return denseFCN.denseFCN_original_instanceNorm(images, is_training,weight_decay,num_classes)

def read_image(image_path, mask_path, image_index):
	imgs = os.listdir(image_path)
	img_name = imgs[image_index]
	mask_name = img_name
	images = io.imread(os.path.join(image_path, img_name))
	image_size = images.shape
	row, col, ch = image_size[0], image_size[1], image_size[2]
	if (ch != 3):
		images = Image.open(os.path.join(image_path, img_name)).convert('RGB')
	# The name for
	if ('PS-boundary' in image_path or 'PS-arbitrary' in image_path):
		mask_name = img_name.replace('ps', 'ms')
		mask_name = mask_name.replace('.jpg', '.png')
	elif ('NIST-2016' in image_path):
		mask_name = img_name.replace('PS', 'MS')
	mask_name = mask_name.replace('.jpg', '.png')

	print(os.path.join(mask_path, mask_name))
	mask = cv2.imread(os.path.join(mask_path, mask_name), 0).astype(dtype=np.uint8)
	mask_copy = np.copy(mask)
	mask[np.where(mask_copy < 128)] = 0
	mask[np.where(mask_copy >= 128)] = 255

	images = np.reshape(images, [1, row, col, 3]).astype(dtype=np.float32) / 255.0
	mask = np.reshape(mask, [1, row, col]).astype(dtype=np.float32) / 255.0
	return images, mask, img_name, mask_name

def read_image_without_mask(image_path,image_index):
	# 2021.11.07
	if test_imgs_txt is not None:
		imgs = []
		with open(test_imgs_txt, 'r') as test_f:
			for line in test_f:
				img = line.strip('\n')
				imgs.append(img)
	else:
		imgs = os.listdir(image_path)
	img_name = imgs[image_index]
	images = io.imread(os.path.join(image_path, img_name))
	image_size = images.shape
	row, col, ch = image_size[0], image_size[1], image_size[2]
	if (ch != 3):
		images = Image.open(os.path.join(image_path, img_name)).convert('RGB')
	# The name for

	if FLAGS.img_size is not None:
		row, col = FLAGS.img_size.split('x')
		print(row, col)
		row, col = int(row), int(col)
		images = images.resize((col, row), Image.ANTIALIAS)

	images = np.reshape(images, [1, row, col, 3]).astype(dtype=np.float32) / 255.0
	return images,img_name


def main(argv=None):
	print_func = print

	shuffle_seed = FLAGS.shuffle_seed
	print_func('Seed={}'.format(shuffle_seed))

	is_training = tf.placeholder(tf.bool, [])
	images = tf.placeholder(tf.float32, [None, None, None, 3])
	imgnames = tf.placeholder(tf.string, [])
	logits_msk, preds_msk, preds_msk_map = model(images, FLAGS.weight_decay, is_training)  # pylint: disable=W0612

	# itr_per_epoch = int(np.ceil(instance_num * FLAGS.train_ratio) / FLAGS.batch_size)
	# print("itr_per_epoch " + str(itr_per_epoch))

	config = tf.ConfigProto(log_device_placement=False)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	saver = tf.train.Saver(max_to_keep=FLAGS.keep_ckpt + 1 if FLAGS.keep_ckpt else 1000000)
	model_checkpoint_path = ''
	if FLAGS.restore and 'ckpt' in FLAGS.restore:
		model_checkpoint_path = FLAGS.restore
	else:
		ckpt = tf.train.get_checkpoint_state(FLAGS.restore or FLAGS.logdir)
		if ckpt and ckpt.model_checkpoint_path:
			model_checkpoint_path = ckpt.model_checkpoint_path
			model_checkpoint_path = model_checkpoint_path.replace('//', '/')

	if model_checkpoint_path:
		try:
			saver.restore(sess, model_checkpoint_path)
		except tf.errors.NotFoundError:  # compatible code
			variables_to_restore = {var.op.name.replace("global_step", "Variable"): var for var in
			                        tf.global_variables()}
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, model_checkpoint_path)
		print_func('Model restored from {}'.format(model_checkpoint_path))

	if FLAGS.mode == 'test':
		warnings.simplefilter('ignore', (UserWarning, RuntimeWarning))

		try:
			image_path = FLAGS.data_dir
			# 2021.11.07
			if test_imgs_txt is not None:
				imgs = []
				with open(test_imgs_txt, 'r') as test_f:
					for line in test_f:
						img = line.strip('\n')
						imgs.append(img)
				imgs.sort()
				test_img_num = len(imgs)
			else:
				test_img_num = FLAGS.test_img_num

			for image_index in tqdm(range(test_img_num)):
				try:
					# print(image_index)
					# images_, labels_msk_, imgnames_, mask_name = read_image(image_path, mask_path, image_index)
					images_, imgnames_ = read_image_without_mask(image_path, image_index)
					logits_msk_, preds_msk_, preds_msk_map_ = sess.run([logits_msk, preds_msk, preds_msk_map],
					                                                   feed_dict={is_training: False, images: images_,
					                                                              imgnames: imgnames_})

					image_shape = images_.shape
					row = image_shape[1]
					col = image_shape[2]

					final_predit_mask_map = np.zeros((row, col))
					final_predit_mask_map += preds_msk_map_[0]

					num_every_pixel_scanned = np.ones((row, col))

					for i in range(FLAGS.batch_size):

						image = np.copy(images_[0])

						rotate_angle = [180]
						recovery_angle = [-180]
						filp_axis = [0, 1]
						save_imgname = str(imgnames_)
						save_imgname = save_imgname.replace('jpg', 'png')
						save_imgname = save_imgname.replace('tif', 'png')
						'''Rotate 180'''
						for angle in range(len(rotate_angle)):
							# print(image.shape)
							test_image = transform.rotate(image, angle=rotate_angle[angle])
							# print(test_image.shape)
							test_image = test_image[np.newaxis, :]
							preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
							                              feed_dict={is_training: False, images: test_image,imgnames: imgnames_})

							final_predit_mask_map += transform.rotate(preds_map_[0], angle=recovery_angle[angle])

							num_every_pixel_scanned += 1
						'''filp'''
						for axis in filp_axis:
							test_image = np.flip(image, axis=axis)
							# test_masks = np.flip(mask, axis=axis)
							test_image = test_image[np.newaxis, :]
							preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
							                              feed_dict={is_training: False, images: test_image,imgnames: imgnames_})

							final_predit_mask_map += np.flip(preds_map_[0], axis=axis)
							num_every_pixel_scanned += 1
						'''Transposed'''
						test_image = np.transpose(image, axes=[1, 0, 2])
						test_image = test_image[np.newaxis, :]
						preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
						                              feed_dict={is_training: False, images: test_image,imgnames: imgnames_})

						print(preds_.min(), preds_.max(), preds_map_.min(), preds_map_.max())

						final_predit_mask_map += np.transpose(preds_map_[0], axes=[1, 0])
						num_every_pixel_scanned += 1

						final_predit_mask_map = final_predit_mask_map / num_every_pixel_scanned

					# save the unthreshold predict maps #
					io.imsave(os.path.join(FLAGS.visout_dir, save_imgname),
					          np.uint8(np.round(final_predit_mask_map * 255.0)))

					preds_msk_ = np.copy(final_predit_mask_map)
					preds_msk_[np.where(final_predit_mask_map <= threshold)] = 0
					preds_msk_[np.where(final_predit_mask_map > threshold)] = 1
					# save the thresholded predict maps #
					io.imsave(os.path.join(FLAGS.visout_threshold_dir, save_imgname),
					          np.uint8(np.round(preds_msk_ * 255.0)))

					# # 2021.9.10
					# plt.subplot(1, 2, 1), plt.imshow(test_image)
					# plt.subplot(1, 2, 2), plt.imshow(np.uint8(np.round(preds_msk_ * 255.0)))
					# ImagePath = '/data/zhengkengtao/Dense-FCN_zhengshu_mix/test_results/{}'.format(save_imgname.replace('.jpg', '.png'))
					# plt.savefig(
					# 	'/data/zhengkengtao/Dense-FCN_zhengshu_mix/test_results/{}'.format(save_imgname.replace('.jpg', '.png')))
					# plt.clf()
					# # ------------------------------------

					print(image_index,save_imgname,file = f)
				except Exception as e:
					print(e)
				# continue
				# print(str(count))

		except tf.errors.OutOfRangeError:
			# break
			print("error")
	else:
		print_func('Mode not defined: ' + FLAGS.mode)
		return None
	f.close()
	# return ImagePath


if __name__ == '__main__':
	tf.app.run()
