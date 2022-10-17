#import du tensorflow et TF_Hub modules
from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed

logging.set_verbosity(logging.ERROR)

# importation des modules necessaires pour la lecture du dataset
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np

# importation des modules necessaires pour l'affichage des animations
import imageio
from IPython import display

from urllib import request  # requires python3

# lien vers l'UCF101 dataset
UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"

# liste contenant les videos dataset
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()

unverified_context = ssl._create_unverified_context()

# fonction pour recuperer tous les videos dans la liste
def list_ucf_videos():
  """Lists videos available in UCF101 dataset."""
  global _VIDEO_LIST
  if not _VIDEO_LIST:
    index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode("utf-8")
    videos = re.findall("(v_[\w_]+\.avi)", index)
    _VIDEO_LIST = sorted(set(videos))
  return list(_VIDEO_LIST)


def fetch_ucf_video(video):
  """Fetchs a video and cache into local filesystem."""
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(UCF_ROOT, video)
    print("Fetching %s => %s" % (urlpath, cache_path))
    data = request.urlopen(urlpath, context=unverified_context).read()
    open(cache_path, "wb").write(data)
  return cache_path

# lecture du fichier video à l'aide de opencv
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

# fonction pour le chargement de la video
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
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

# transformation du video en fichier .gif si necessaire
def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=25)
  return embed.embed_file('./animation.gif')

# Recuperation du kinetics-400 labels d'action depuis le repo git 
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
  labels = [line.decode("utf-8").strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))

# recuperer la liste des videos dans le dataset
ucf_videos = list_ucf_videos()

#c grouper les videos pour chaque categories
categories = {}
for video in ucf_videos:
  category = video[2:-12]
  if category not in categories:
    categories[category] = []
  categories[category].append(video)
print("Found %d videos in %d categories." % (len(ucf_videos), len(categories)))

for category, sequences in categories.items():
  summary = ", ".join(sequences[:2])
  print("%-20s %4d videos (%s, ...)" % (category, len(sequences), summary))

# video_path = fetch_ucf_video("v_CricketShot_g04_c02.avi")
# url du video à reconnaitre
video_path = "video_test/guitar-playing.gif"
sample_video = load_video(video_path)[:100]
sample_video.shape

# chargement du modèle Convnet 3D gonflé pré-entraîné pour la reconnaissance d'actions sur Kinetics-400
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

# fonction de prediction du video en paramètre
def predict(sample_video):
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

  logits = i3d(model_input)['default'][0]
  probabilities = tf.nn.softmax(logits)

  # recupération de la meilleure prédiction d'action
  idx_predict = np.argsort(probabilities)[::-1][:1]
  lbl = str.format(labels[int(idx_predict)])

  show_video(video_path, lbl)

# affichage du résultat à l'aide d'opencv
def show_video(video, label):
  vid_capture = cv2.VideoCapture(video)

  while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if ret == True:
      cv2.putText(frame, label, (50,50), font, 1, (0, 255, 355), 2, cv2.LINE_4)
      cv2.imshow('Human recognition activity', frame)
      key= cv2.waitKey(20)
      if key == ord('q'):
        break
    else:
      vid_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
      continue
    
predict(sample_video)