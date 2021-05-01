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
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # Design lines on the hand connecting points
        return img

    # def findPosition(self, img, handNo, draw=True):
    #     lmlist = []
    #     for id, lm in enumerate(handLms.landmark):
    #         h, w, c = img.shape
    #         cx, cy = int(lm.x * w), int(lm.y * h)
    #         print(id, cx, cy)
    #         # if id == 5:  # Id of the finger
    #         # cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED) # design circle on the selected point
    #     return lmlist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Choose Camera device
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Img", img)  # Show Camera's recording in window
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
