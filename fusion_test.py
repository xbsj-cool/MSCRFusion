import os
import torch
from torch.autograd import Variable
from fusion.net import autoencoder
from fusion import utils
from fusion.fusion_strategy import IMV_F
from fusion.args_fusion import args
import numpy as np
import cv2
import time

def load_model(path):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = autoencoder(nb_filter, input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('\nModel {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))
	nest_model.eval()

	if args.cuda:
		nest_model.cuda()
	return nest_model

processing_times = []
def run_demo(nest_model, infrared_path, visible_path, output_path_root, file_name, ir_mask_path, vi_mask_path, bg_mask_path):
	# start_time = time.time()
	img_ir = utils.get_test_image(infrared_path)
	img_vi = utils.get_test_image(visible_path)
	img_ir_mask = utils.get_test_image(ir_mask_path)
	img_ir_mask = img_ir_mask/255 # Binary mask, the value range is [0,1]
	img_vi_mask = utils.get_test_image(vi_mask_path)
	img_vi_mask = img_vi_mask/255
	img_bg_mask = utils.get_test_image(bg_mask_path)
	img_bg_mask = img_bg_mask/255

	if args.cuda:
		img_ir = img_ir.cuda()
		img_vi = img_vi.cuda()
		img_ir_mask = img_ir_mask.cuda()
		img_vi_mask = img_vi_mask.cuda()
		img_bg_mask = img_bg_mask.cuda()

	img_ir = Variable(img_ir, requires_grad=False)
	img_vi = Variable(img_vi, requires_grad=False)
	img_ir_mask = Variable(img_ir_mask, requires_grad=False)
	img_vi_mask = Variable(img_vi_mask, requires_grad=False)
	img_bg_mask = Variable(img_bg_mask, requires_grad=False)
	inf_vis_salient_background = IMV_F().Inf_Vis_Salient_Background

	img_ir_truth, img_ir_back, img_vi_truth, img_vi_back = inf_vis_salient_background(img_ir, img_vi, img_ir_mask, img_vi_mask, img_bg_mask)

	en_ir_truth = nest_model.encoder(img_ir_truth)
	en_ir_back = nest_model.encoder(img_ir_back)

	en_vi_truth = nest_model.encoder(img_vi_truth)
	en_vi_back = nest_model.encoder(img_vi_back)

	# fusion
	f = nest_model.IMV_F_fusion_strategy(en_ir_truth, en_ir_back, en_vi_truth, en_vi_back)

	# decoder
	img_fusion = nest_model.decoder_eval(f)

	output_path = output_path_root + file_name
	# save images
	utils.save_image_test(img_fusion, output_path)
	# end_time = time.time()
	# processing_time = end_time - start_time
	# processing_times.append(processing_time)
	# print(f"Processing time: {processing_time:.4f} seconds")


def fuse_main(epoch, ssim_path_id, test_path):
	# run demo
	test_path = test_path + 'ir'
	file_name = os.listdir(test_path)

	model_path_file = args.model_default


	epoch_num = epoch + 1
	model_name = 'final' + '.model'
	epoch_model_path = os.path.join(model_path_file, model_name)
	with torch.no_grad():
		model_path = epoch_model_path
		model = load_model(model_path)

		output_path = './1/'
		if os.path.exists(output_path) is False:
			os.mkdir(output_path)

		output_path = output_path + '/'
		print('Processing......fuse...  ' + 'epoch_' + str(epoch_num))

		for i in range(len(file_name)):
			infrared_path = os.path.join(test_path, file_name[i])
			visible_path_, ir_mask_path_, vi_mask_path_, bg_mask_path_ = test_path.split('/'), test_path.split('/'), test_path.split('/'), test_path.split('/')

			if visible_path_[-1] is '':
				visible_path_[-2] = 'vi'
				ir_mask_path_[-2] = 'ir_mask'
				vi_mask_path_[-2] = 'vi_mask'
				bg_mask_path_[-2] = 'fuse_mask'
			else:
				visible_path_[-1] = 'vi'
				ir_mask_path_[-1] = 'ir_mask'
				vi_mask_path_[-1] = 'vi_mask'
				bg_mask_path_[-1] = 'fuse_mask'

			visible_path = os.path.join('/'.join(visible_path_), file_name[i])
			ir_mask_path = os.path.join('/'.join(ir_mask_path_), file_name[i])
			vi_mask_path = os.path.join('/'.join(vi_mask_path_), file_name[i])
			bg_mask_path = os.path.join('/'.join(bg_mask_path_), file_name[i])

			run_demo(model, infrared_path, visible_path, output_path, file_name[i], ir_mask_path, vi_mask_path, bg_mask_path)

	average_time = np.mean(processing_times)
	print(f"Average processing time: {average_time:.4f} seconds")
	print('Done......')


if __name__ == '__main__':
	test_path = './image/'
	fuse_main(0, 2, test_path)