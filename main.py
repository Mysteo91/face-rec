import time
import torch
import mqtt
from facenet_pytorch import MTCNN

import face_recognition
import os
import cv2
import numpy as np
import requests
from urllib3.util import url
import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()



KNOWN_FACES_DIR = 'KnownFaces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.53
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
TIME_FOR_REPEAT = 1.0

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower()) - 97) * 8 for c in name[:3]]
    return color


mqtt_client = mqtt.run()

print('Loading known faces...')
known_faces = []
known_names = []

# for name in os.listdir(KNOWN_FACES_DIR):
#
#     # Next we load every file of faces of known person
#     for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
#         # Load an image
#         image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
#         # Get 128-dimension face encoding
#         # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
#         encoding = face_recognition.face_encodings(image)[0]
#
#         # Append encodings and name
#         known_faces.append(encoding)
#         known_names.append(name)

################################################# PREPARE OPEN CV#################################


# print('Processing unknown faces...')
# # Now let's loop over a folder of faces we want to label
# for filename in os.listdir(UNKNOWN_FACES_DIR):
#     print(f'Filename {filename}', end='')
#     image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')



# Capture frame-by-frame
cap = VideoCapture("rtsp://rtsp:aK973pS44@192.168.0.55:554/av_stream/ch0")
#cap = VideoCapture(0)
start_frame_number = 1
timing = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
while True:
    frame = cap.read()
    frame = cv2.resize(frame, (320, 240))
    image = cap.read()
    image = cv2.resize(image, (324, 240))

    boxes, _ = mtcnn.detect(image)
    if not (boxes is None):
        for box in boxes:
            cv2.rectangle(image, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (255, 255, 0))
    cv2.imshow('rez', image)

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(frame, model=MODEL)
    if len(locations) > 0:
        cv2.rectangle(frame, (locations[0][3], locations[0][2]), (locations[0][1], locations[0][0]), (255, 255, 0))
    cv2.imshow('rez2', frame)
    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    #encodings = face_recognition.face_encodings(image, locations)
    encodings = []

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people

    # if time.time() - timing > TIME_FOR_REPEAT:
    #     mqtt.publish(mqtt_client, 'online', 'camera/online')
    #     TIME_FOR_REPEAT = 0.4
    #     timing = time.time()
    #     print(f', found {len(encodings)} face(s)')
    #     if len(encodings) == 0:
    #         mqtt.publish(mqtt_client, 'no', 'camera/match')
    #     else:
    #         for face_encoding, face_location in zip(encodings, locations):
    #
    #             # We use compare_faces (but might use face_distance as well)
    #             # Returns array of True/False values in order of passed known_faces
    #             # results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
    #             results = face_recognition.face_distance(known_faces, face_encoding)
    #             best_match_index = np.argmin(results)
    #             match = None
    #             if results[best_match_index] <= TOLERANCE:
    #                 match = known_names[best_match_index]
    #                 mqtt.publish(mqtt_client, 'yes', 'camera/match')
    #                 mqtt.publish(mqtt_client, match, 'camera/face')
    #                 TIME_FOR_REPEAT = 3
    #
    #             else:
    #                 match = "Unknown"
    #                 mqtt.publish(mqtt_client, 'no', 'camera/match')
    #                 print(f' - {match} from UNKNOWN')
    #                 print(face_location)




    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        # cv2.destroyWindow('webcam'+str(webcam))
        cv2.destroyAllWindows()
        print("We all done ")
