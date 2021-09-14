"""
 > Script for testing .pth models  
    * set model_name ('DPGAN') and  model path
    * set data_dir (input) and sample_dir (output) 
"""
# py libs
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2
## options
from utils.data_utils import LoadImages

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/test/exp/validation/")
parser.add_argument("--sample_dir", type=str, default="data/OurMeth")
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
parser.add_argument("--model_path", type=str, default="checkpoints/DPGAN/underwater_type4/generator_120.pth")
opt = parser.parse_args()

## checks
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch
if opt.model_name.lower()=='funiegan':
    from nets import dpgan
    model = dpgan.GeneratorDPGAN()
else: 
    # other models
    pass

## load weights
model.load_state_dict(torch.load(opt.model_path))
if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 1280, 704, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)


## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))

device = torch.device('cuda:0')

save_img = True
vid_path, vid_writer = None, None
imgsz = 480
dataset = LoadImages(opt.data_dir, img_size=imgsz)
for path, img, im0s, vid_cap in dataset:
    # img = Variable(img).type(Tensor).unsqueeze(0)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    gen_img = model(img)
    # times.append(time.time() - s)
    # save output
    img_sample = torch.cat((img.data, gen_img.data), -1)
    # save_image(gen_img.data, join(opt.sample_dir, basename(path)), normalize=True)
    save_path = join(opt.sample_dir, basename(path))
    img = gen_img.data.to(torch.device('cpu'))
    input_tensor = img.squeeze()
    # from [0,1] To [0,255]
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB T0 BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    view_img =True
    if view_img:
        cv2.imshow("demo", input_tensor)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration
    if save_img:
        if dataset.mode == 'images':
            cv2.imwrite(save_path, input_tensor)
        else:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                # fps = 30
                # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = 960
                h = 288
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
            vid_writer.write(input_tensor)
        # vid_writer.write(gen_img.data)
    print("Tested: %s" % path)

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))



