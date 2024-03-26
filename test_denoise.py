import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from model import *
from utils import *
from keras import Model, Input
from PIL.Image import fromarray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json


def evaluate(config, model):
    results = []
    for name in os.listdir(config.test_path):
        fullname = os.path.join(config.test_path, name)
        lr = cv2.imread(fullname)
        ft = cv2.imread(fullname)
        lr = np.array(get_lowres_image(fromarray(lr), mode='denoise'))
        ft = np.array(get_lowres_image(fromarray(ft), mode='denoise'))

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        ft = cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
        
        out = predict_images(model, lr)
        
        # Compute PSNR
        psnr_score = psnr(ft, np.array(out), data_range=np.max(ft) - np.min(ft))
        
        # Compute SSIM
        # ssim_score = ssim(ft, np.array(out), multichannel=True, data_range=np.max(ft) - np.min(ft))
        
        result = {
            "image_name": name,
            "PSNR": psnr_score,
            # "SSIM": ssim_score
        }
        results.append(result)
    
    # Save results to JSON file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f)

    avg_psnr = np.mean([result["PSNR"] for result in results])
    # avg_ssim = np.mean([result["SSIM"] for result in results])
    print("Average PSNR:", avg_psnr)
    # print("Average SSIM:", avg_ssim)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

	# Input Parameters
    parser.add_argument('--test_path', type=str, default="CNN/MIRNet-Keras/withGAN/dataset_complete/testing")
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--checkpoint_filepath', type=str, default="CNN/MIRNet-Keras/weights_new/denoise_without_Gan/bestEpochTillNow/")
    parser.add_argument('--num_rrg', type=int, default=3)
    parser.add_argument('--num_mrb', type=int, default=2)
    parser.add_argument('--num_channels', type=int, default=64)

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    mri_x = MIRNet(config.num_channels, config.num_mrb, config.num_rrg)
    x = Input(shape=(None, None, 3))
    out = mri_x.main_model(x)
    model = Model(inputs=x, outputs=out)

    model.load_weights(config.checkpoint_filepath + 'best_till_now.h5')

    evaluate(config, model)
