import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
import copy
from util import *
from PIL import Image

from model import *
import moviepy.video.io.ImageSequenceClip
import scipy
import cv2
import dlib
import kornia.augmentation as K
from aubio import tempo, source

from IPython.display import HTML
from base64 import b64encode
import gradio as gr

device = 'cpu'
latent_dim = 8
n_mlp = 5
num_down = 3

G_A2B = Generator(256, 4, latent_dim, n_mlp, channel_multiplier=1, lr_mlp=.01,n_res=1).to(device).eval()

ensure_checkpoint_exists('GNR_checkpoint.pt')
ckpt = torch.load('GNR_checkpoint.pt', map_location=device)

G_A2B.load_state_dict(ckpt['G_A2B_ema'])

# mean latent
truncation = 1
with torch.no_grad():
    mean_style = G_A2B.mapping(torch.randn([1000, latent_dim]).to(device)).mean(0, keepdim=True)


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
])

def inference(input_im):
        mode = 'beat'
        assert mode in ('normal', 'blend', 'beat', 'eig')


        # Frame numbers and length of output video
        start_frame=0
        end_frame=None
        frame_num = 0
        mp4_fps= 30
        faces = None
        smoothing_sec=.7
        eig_dir_idx = 1 # first eig isnt good so we skip it

        frames = []
        reader = cv2.VideoCapture(inpath)
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # get beats from audio
        win_s = 512                 # fft size
        hop_s = win_s // 2          # hop size

        s = source(inpath, 0, hop_s)
        samplerate = s.samplerate
        o = tempo("default", win_s, hop_s, samplerate)
        delay = 4. * hop_s
        # list of beats, in samples
        beats = []

        # total number of frames read
        total_frames = 0
        while True:
            samples, read = s()
            is_beat = o(samples)
            if is_beat:
                this_beat = int(total_frames - delay + is_beat[0] * hop_s)
                beats.append(this_beat/ float(samplerate))
            total_frames += read
            if read < hop_s: break
        #print len(beats)
        beats = [math.ceil(i*mp4_fps) for i in beats]


        if mode == 'blend':
            shape = [num_frames, 8, latent_dim] # [frame, image, channel, component]
            all_latents = random_state.randn(*shape).astype(np.float32)
            all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps, 0, 0], mode='wrap')
            all_latents /= np.sqrt(np.mean(np.square(all_latents)))
            all_latents = torch.from_numpy(all_latents).to(device)
        else:
            all_latents = torch.randn([8, latent_dim]).to(device)

        if mode == 'eig':
            all_latents = G_A2B.mapping(all_latents)

        in_latent = all_latents

        # Face detector
        face_detector = dlib.get_frontal_face_detector()

        assert start_frame < num_frames - 1
        end_frame = end_frame if end_frame else num_frames

        while reader.isOpened():
            _, image = reader.read()
            if image is None:
                break

            if frame_num < start_frame:
                continue
            # Image size
            height, width = image.shape[:2]

            # 2. Detect with dlib
            if faces is None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_detector(gray, 1)
            if len(faces):
                # For now only take biggest face
                face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_face = Image.fromarray(cropped_face)
            frame = test_transform(cropped_face).unsqueeze(0).to(device)

            with torch.no_grad():
                A2B_content, A2B_style = G_A2B.encode(frame)
                if mode == 'blend':
                    in_latent = all_latents[frame_num]
                elif mode == 'normal':
                    in_latent = all_latents
                elif mode == 'beat':
                    if frame_num in beats:
                        in_latent = torch.randn([8, latent_dim]).to(device)

                if mode == 'eig':
                    if frame_num in beats:
                        direction = 3 * eigvec[:, eig_dir_idx].unsqueeze(0).expand_as(all_latents).to(device)
                        in_latent = all_latents + direction
                        eig_dir_idx += 1

                    fake_A2B = G_A2B.decode(A2B_content.repeat(8,1,1,1), in_latent, use_mapping=False)
                else:
                    fake_A2B = G_A2B.decode(A2B_content.repeat(8,1,1,1), in_latent)



                fake_A2B = torch.cat([fake_A2B[:4], frame, fake_A2B[4:]], 0)

                fake_A2B = utils.make_grid(fake_A2B.cpu(), normalize=True, range=(-1, 1), nrow=3)


            #concatenate original image top
            fake_A2B = fake_A2B.permute(1,2,0).cpu().numpy()
            frames.append(fake_A2B*255)

            frame_num += 1

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=mp4_fps)

        # save to temporary file. hack to make sure ffmpeg works
        #clip.write_videofile('./temp.mp4')

        # use ffmpeg to add audio to video
        #!ffmpeg -i ./temp.mp4 -i $inpath -c copy -map 0:v:0 -map 1:a:0 $outpath -y
        #!rm ./temp.mp4
        return outpath


gr.Interface(
    inference, 
    [gr.inputs.Video(type=None, label="Input")], 
    gr.outputs.Video(label="Output"),
    ]).launch()
