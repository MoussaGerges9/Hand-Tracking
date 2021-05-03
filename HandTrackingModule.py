import cv2
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon,
                                        self.trackCon)  # Takes max hands and confidence
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # Design lines on the hand connecting points
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bBox = []

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmlist.append([id, cx, cy])
                # if id == 5:  # Id of the point
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # design circle on the selected point

            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bBox = xMin, yMin, xMax, yMax

            if draw:
                cv2.rectangle(img, (bBox[0] - 20, bBox[1] - 20), (bBox[2] + 20, bBox[3] + 20), (255, 0, 0),
                              2)  # design rectangle around the hand

        return lmlist, bBox


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Choose Camera device
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if (len(lmList) != 0):
            print(lmList[5])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Img", img)  # Show Camera's recording in window
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
