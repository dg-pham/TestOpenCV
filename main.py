import os
import time
import hand
import cv2
# import numpy as np

cap = cv2.VideoCapture(0)

pTime = 0

path = 'Fingers'
lst = os.listdir(path)
lst2 = []

for i in lst:
    image = cv2.imread(f'{path}/{i}')
    lst2.append(image)

detector = hand.handDetector(detectionCon=0.55)
fingerID = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame, draw=False)

    fingers = []

    # Algorithm
    if len(lmlist) != 0:

        # thumb
        if lmlist[fingerID[0]][1] < lmlist[fingerID[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # index finger, middle finger, ring finger, little finger
        for id in range(1, 5):
            if lmlist[fingerID[id]][2] < lmlist[fingerID[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

    # results
    fingersCount = fingers.count(1)
    cv2.rectangle(frame, (0, 200), (150, 400), (0, 0, 0), 7)
    cv2.putText(frame, str(fingersCount), (28, 350), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 5, (203, 192, 255), 4)

    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 192, 0), 2)

    # show fingers images
    h, w, c = lst2[fingersCount - 1].shape
    frame[0:h, 0:w] = lst2[fingersCount - 1]

    # # multiple cameras
    # width = int(cap.get(3))
    # height = int(cap.get(4))
    #
    # smail_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # image = np.zeros(frame.shape, np.uint8)
    #
    # image[:height // 2, :width // 2] = smail_frame
    # image[:height // 2, width // 2:] = cv2.rotate(smail_frame, cv2.ROTATE_180)
    # image[height // 2:, :width // 2] = cv2.rotate(smail_frame, cv2.ROTATE_180)
    # image[height // 2:, width // 2:] = smail_frame

    cv2.imshow('', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
