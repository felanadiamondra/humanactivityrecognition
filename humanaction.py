from absl import logging #Tensorflow and TF-Hub modules
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
logging.set_verbosity(logging.ERROR)

# importing som modules to help reading the UCF101 dataset
import random
import re
import tempfile
import ssl
import cv2
import numpy as np

# importing some modules to display an animation using imageio
import imageio
from IPython import display
from urllib import request # requires python3

# Fonction d'assistance pour l'ensemble de données UCF101
UCF_ROOT  = "https://www.crccv.ucf.edu/THUMOS14/UCFUCF101/UCF101/"
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()

unverified_context = ssl._create_unverified_context()

def list_ucf_videos():
    """Lists videos available in UCF101 dataset"""
    global _VIDEO_LIST
    if not _VIDEO_LIST:
        index = request.urlopen(UCF_ROOT, context = unverified_context).read().decode("utf-8")
        videos = re.findall("((v_[\w_]+\.avi))", index)
        _VIDEO_LIST = sorted(set(videos))
    return list(_VIDEO_LIST)

def fetch_ucf_video(video):
    """ Fetchs a video and cache into local filesystem """
    cache_path = os.path.join(_CACHE_DIR, video)
    if not os.path.exists(cache_path):
        urlpath = request.urljoin(UCF_ROOT, video)
        print("Fetchin %s => %s" (urlpath, cache_path))
        data = request.urlopen(urlpath, context= unverified_contect).read()
        open(cache_path, "wb").write(data)
    return cache_path

#Utilities to open video files using CV2
def crop_center_square(frame):
    y,x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x //2) - (min_dim // 2)
    start_y = (y//2) - (min_dim //2)
    return frame[start_y : start_y + min_dim, start_x:start_x + min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2,1,0]]

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0

def to_gif(images):
    converted_images = np.clip(images * 255.0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=25)
    return embed.embed_file('./animation.gif')

KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
  labels = [line.decode("utf-8").strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))