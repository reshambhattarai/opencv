import cv2
import mediapipe as mp
import time

#take video capture from webcam in my usb cam . use 0 if its ur default webcam
cap = cv2.VideoCapture(1)

#calling out hands detection object from mediapipe. solutions is just a mediapipe protocol
mpHands =mp.solutions.hands
hands=mpHands.Hands()

#calling out drawing tool from mediapipe
mpDraw = mp.solutions.drawing_utils

#setting default value of previous frame number and current frame number to 0 for future frame counts
pTime = 0
cTime = 0

#since its a video caputre, we use loop to use each frame as an image
while True:
	#storing a frame in img. since cap.read() gives two values, success is just a default parameter
	success, img = cap.read()
	
	#converting our BGR image to RGB. since opencv uses BGR mode to process while mediapipe uses RGB 
	imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	
	#calling out process function from hands object of mediapipe. it looks out for landmarks in hands
	results= hands.process(imgRGB)
	
	#print(results.multi_hand_landmarks)
	
	#if landmarks available then draw the landmarks
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			#defining the landmarks with their corresponding id an assigining pixel value
			for id, lm in enumerate(handLms.landmark):
				#print(id,lm)
				h,w,c=img.shape
				cx,cy=int(lm.x*w),int(lm.y*h)
				print(id,cx,cy)
			#drawing landmarks
			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
	
	#frame count and display fps
	cTime = time.time()
	fps= 1/(cTime-pTime)
	pTime = cTime
	
	cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
	
	#showing everything in a box
	cv2.imshow("Image",img)
	
	cv2.waitKey(1)
