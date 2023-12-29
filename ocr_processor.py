import cv2
import numpy as np
import easyocr
import torch

def text_recognition(img, reader):
  image = img
  words = []
  boxes = []

  brightness = 100
  contrast = 100
  img = np.int16(image)
  img = img * (contrast / 127 + 1) - contrast + brightness
  img = np.clip(img, 0, 255)
  img = np.uint8(img)

  result = reader.readtext(img, detail=1, width_ths=0.1, height_ths=1, ycenter_ths=0.5, text_threshold=0.4)

  for (coord, text, prob) in result:
      normal_coords = []
      words.append(text)
      (topleft, topright, bottomright, bottomleft) = coord
      normal_coords.append(topleft[0])
      normal_coords.append(topleft[1])
      normal_coords.append(bottomright[0])
      normal_coords.append(bottomright[1])
      boxes.append(normal_coords)
      # tx, ty = (int(topleft[0]), int(topleft[1]))
      # bx, by = (int(bottomright[0]), int(bottomright[1]))
      # cv2.rectangle(img, (tx, ty), (bx, by), (0, 0, 255), 2)

  # cv2.imwrite("savedImage.jpg", img)

  boxes = torch.tensor(boxes)

  return words, boxes