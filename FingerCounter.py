import cv2
import handTrackingModule as htm
import time
import os

#Constants
tipIds = [4, 8, 12, 16, 20]#Id of each of the finger tip
####################################
#Variables
cameraNo = 0
wCam, hCam = 640, 480
wFingerImg, hFingerImg = 200, 200
pastTime = 0
currentTime = 0
colorMagenta = (255, 0, 255)
colorGreen = (150, 200, 0)
folderPath = "ImagesFingers"
####################################

cap = cv2.VideoCapture(cameraNo)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionConfidence=0.75)

myList = os.listdir(folderPath)
#print(myList)
listOfFingersImgs = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    image = cv2.resize(image, (wFingerImg, hFingerImg))
    listOfFingersImgs.append(image)

print(f'Number of images in the folder: {len(listOfFingersImgs)}')

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []
        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:#by using x pos of the thumb tip
            fingers.append(1)
        else:
            fingers.append(0)
        #Other 4 fingers
        for index in range(1, 5):
            if lmList[tipIds[index]][2] < lmList[tipIds[index] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)#returns how many 1 in the fingers list
        #print(totalFingers)
        h, w, c = listOfFingersImgs[totalFingers - 1].shape
        img[0:h, 0:w] = listOfFingersImgs[totalFingers - 1]#if -1 then returns the last item of the list so it is 6th img

        cv2.rectangle(img, (25, 225), (175, 425), colorGreen, cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, colorMagenta, 25)

    #fps calculation
    currentTime = time.time()
    fps = 1 / (currentTime - pastTime)
    pastTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (450, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                colorMagenta, 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break