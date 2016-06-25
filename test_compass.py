#!/usr/bin/env python

from VisualCompass import VisualCompass
import cv2
from math import pi

vc = VisualCompass([ "image1.png" ])
img = cv2.imread("image2.png",0)
angle = vc.getNorth(img)
print angle * 180.0/pi,"degrees"

