import os
import math
import argparse
import numpy as np
import tensorflow as tf

from model import MIRNet
from tensorflow import keras
from IPython.display import display
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.optimizers.legacy import Adam
from utils import *
from keras.models import load_model


# print(os.getcwd())


train_ds = SSID(subset='train').dataset(repeat_count=1)
valid_ds = SSID(subset='valid').dataset(repeat_count=1)

# test_path = 'CNN/MIRNet-Keras/dataset_polarimetric_output/test/PARAM_POLAR' # works
# test_path = './drive/MyDrive/all_dataset/dataset_polarimetric_output/test/PARAM_POLAR' # works
test_path = "CNN/MIRNet-Keras/withGAN/dataset_complete/testing"
# test_path = 'CNN/MIRNet-Keras/dataset_polarimetric_output/test/PARAM_POLAR'
# test_img_paths = sorted(
#     [
#         os.path.join(test_path, fname)
#         for fname in os.listdir(test_path)
#         if fname.endswith(".PNG")
#     ]
# )



test_img_paths = []

for i in os.listdir(test_path):
    if i.endswith(".png"):
        test_img_paths.append(os.path.join(test_path, i))

# print(test_img_paths, 'check2')


def plot_epoch_metrics(epoch_psnr, epoch_loss):
    plt.figure(figsize=(12, 6))

    # Plotting PSNR
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_psnr) + 1), epoch_psnr, label='PSNR', marker='o')
    plt.title('Epoch vs PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()

    # Plotting Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, label='Loss', marker='o', color='orange')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()        

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    mir_x = MIRNet(64, config.num_mrb, config.num_rrg)
    x = Input(shape=(None, None, 3))
    out = mir_x.main_model(x)
    model = Model(inputs=x, outputs=out)
    
    
    early_stopping_callback = EarlyStopping(monitor="val_psnr_denoise", patience=10, mode='max')
    checkpoint_filepath = config.checkpoint_filepath

    current_epoch_callback = ModelCheckpoint(
        checkpoint_filepath + 'currentEpoch/' + f'epoch_{{epoch:02d}}.h5',
        save_every='epoch',  # Save after each epoch
        verbose=2
    )
    best_epoch_callback = ModelCheckpoint(
        checkpoint_filepath + 'bestEpochTillNow/'+ f'best_{{val_psnr_denoise:.2f}}.h5',
        monitor='val_psnr_denoise',
        mode='max',
        save_best_only=True,
        verbose=2
    )

    
    # json_file_path = 'CNN/MIRNet-Keras/weights/json_file' #works
    json_file_path = './mirnet-denoising/weights/json_file' #works
    callbacks = [ESPCNCallback(test_img_paths, mode=config.mode, checkpoint_ep=config.checkpoint_ep, json_file_path=json_file_path), early_stopping_callback, best_epoch_callback]
    loss_fn = MeanSquaredError()
    optimizer = Adam(learning_rate = config.lr)

    epochs = 15 #config.num_epochs

    model.compile(
        optimizer=optimizer, loss=loss_fn, metrics=[psnr_denoise, ssim_denoise]
    )

    history = model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=1,
    )



if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()

	# # Input Parameters
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--checkpoint_ep', type=int, default=2)
	# parser.add_argument('--checkpoint_filepath', type=str, default="CNN/MIRNet-Keras/weights/denoise/")
	parser.add_argument('--checkpoint_filepath', type=str, default="CNN/MIRNet-Keras/weights/denoise/bestEpochTillNow/")
	parser.add_argument('--num_rrg', type=int, default= 3)
	parser.add_argument('--num_mrb', type=int, default= 2)
	parser.add_argument('--mode', type=str, default= 'denoise')

	config = parser.parse_args()

	if not os.path.exists(config.checkpoint_filepath):
		os.mkdir(config.checkpoint_filepath)

	train(config)