import os
import cv2
import argparse
import numpy as np
import tensorflow as tf 

from model import *
from utils import *
from keras import Model, Input
from PIL.Image import fromarray


def run(config, model):
    for name in os.listdir(config.test_path):
        fullname = os.path.join(config.test_path, name)
        lr = cv2.imread(fullname)
        ft = cv2.imread(fullname)
        # print(fullname, 'test path')
        lr = np.array(get_lowres_image(fromarray(lr), mode='denoise'))
        ft = np.array(get_lowres_image(fromarray(ft), mode='denoise'))

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        ft = cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
        out = predict_images(model, lr)
        print(name, 'name')
        print(psnr_denoise(ft, np.array(out)), 'psnr')

        # print(os.path.join(fullname.replace('test', 'result')), 'print')
        cv2.imwrite(os.path.join(fullname.replace('test', 'result')), cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

	# Input Parameters
    parser.add_argument('--test_path', type=str, default="CNN/MIRNet-Keras/dataset_polarimetric_modified/test/images")
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--checkpoint_filepath', type=str, default="CNN/MIRNet-Keras/weights/denoise/")
    parser.add_argument('--num_rrg', type=int, default= 3)
    parser.add_argument('--num_mrb', type=int, default= 2)
    parser.add_argument('--num_channels', type=int, default= 64)

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    mri_x = MIRNet(config.num_channels, config.num_mrb, config.num_rrg)
    x = Input(shape=(None, None, 3))
    out = mri_x.main_model(x)
    model = Model(inputs=x, outputs=out)
    

    # print(config.checkpoint_filepath, 'file path')
    model.load_weights(config.checkpoint_filepath + 'MIR_Denoise.h5')

    run(config, model)
