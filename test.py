# import argparse
# import time

# import torch
# from PIL import Image
# from torch.autograd import Variable
# from torchvision.transforms import ToTensor, ToPILImage

# from model import Generator

# parser = argparse.ArgumentParser(description='Test Single Image')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
# parser.add_argument('--image_name', type=str, help='test low resolution image name')
# parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
# opt = parser.parse_args()

# UPSCALE_FACTOR = opt.upscale_factor
# TEST_MODE = True if opt.test_mode == 'GPU' else False
# IMAGE_NAME = opt.image_name
# MODEL_NAME = opt.model_name

# model = Generator(UPSCALE_FACTOR).eval()
# if TEST_MODE:
#     model.cuda()
#     model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
# else:
#     model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

# image = Image.open(IMAGE_NAME)
# image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
# if TEST_MODE:
#     image = image.cuda()

# start = time.time()
# out = model(image)
# elapsed = (time.time() - start)
# print('cost' + str(elapsed) + 's')
# out_img = ToPILImage()(out[0].data.cpu())
# out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)


import argparse
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Images in a Folder')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--input_folder', type=str, help='input folder containing low resolution images')
parser.add_argument('--output_folder', type=str, help='output folder to save high resolution images')
parser.add_argument('--model_name', default='netG_epoch_4_75.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
INPUT_FOLDER = opt.input_folder
OUTPUT_FOLDER = opt.output_folder
MODEL_NAME = opt.model_name

# Create output folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

# Get list of image files in the input folder
image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]

for image_name in image_files:
    image_path = os.path.join(INPUT_FOLDER, image_name)
    image = Image.open(image_path)
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.time()
    out = model(image)
    elapsed = (time.time() - start)
    print(f'Image {image_name} processed in {elapsed:.4f} seconds')
    
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(os.path.join(OUTPUT_FOLDER, image_name))
