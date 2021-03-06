import math
import time
from ctypes import cast, POINTER

import cv2
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pynput.keyboard import Controller, KeyCode

import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)  # Choose Camera device
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
minVol = volumeRange[0]
maxVol = volumeRange[1]

videoState = True
playPause = True

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    # print(lmList)

    if len(lmList) != 0:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100  # size of the box around the hand

        fingers = detector.fingersUp(lmList)
        downFingers = fingers[2] + fingers[3] + fingers[4]
        allFingers = downFingers + fingers[0] + fingers[1]

        if 150 < area < 1000 and downFingers == 0 and fingers[1] != 0:  # filter by the size and control other fingers

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Get middle point value

            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)  # middle point

            length = math.hypot(x2 - x1, y2 - y1)  # Get the length

            # Change Volume
            volBar = np.interp(length, [30, 200], [minVol, maxVol])
            volPer = np.interp(length, [30, 200], [0, 100])
            # volume.SetMasterVolumeLevel(volBar, None)

            smoothness = 5
            volPer = smoothness * round(volPer / smoothness)
            volume.SetMasterVolumeLevelScalar(volPer / 100, None)  # More accurate

            if length < 30:
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)  # middle point

            elif length > 200:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)  # middle point

        elif allFingers == 0 and playPause:  # Pause video
            videoState = True
            playPause = False

        elif allFingers == 5 and not playPause:  # Play video
            videoState = True
            playPause = True

        elif videoState:  # Play and Pause button
            keyboard = Controller()
            keyboard.press(KeyCode.from_vk(0xB3))
            videoState = False

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime - 1
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)  # Show Camera's recording in window
    cv2.waitKey(1)
