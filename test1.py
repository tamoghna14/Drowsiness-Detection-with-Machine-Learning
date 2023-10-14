from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
#from pygame import mixer
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

def alarm(msg):
	global alarm_status
	global alarm_status2
	global saying
	
	while alarm_status:
		print('drowsiness call')
		s = 'espeak "'+msg+'"'
		os.system(s)
	if alarm_status2:
		print('yawning call')
		saying=True
		s = 'espeak "'+msg+'"'
		os.system(s)
		saying=False
		

def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	C = euclidean_dist(eye[0], eye[3])
	
	ear = (A + B) / (2.0 * C)
	return ear
	
def lip_distance(shape):
	top_lip = shape[50:53]
	top_lip = np.concatenate((top_lip, shape[61:64]))
	
	low_lip = shape[56:59]
	low_lip = np.concatenate((low_lip, shape[65:68]))
	
	top_mean = np.mean(top_lip, axis=0)
	low_mean = np.mean(low_lip, axis=0)
	
	distance = abs(top_mean[1] - low_mean[1])
	return distance

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
#ap.add_argument("-a", "--alarm", type=int, default=0,
#	help="boolean used to indicate if TrafficHat should be used")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.29
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 40
COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False
#mixer.init()
#sound = mixer.Sound('alarm.wav')


print("---> Loading facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("---> Starting video stream....")
vs = VideoStream(src=0).start()

time.sleep(1.0)

# loop over frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	# loop over the face detections
	for (x, y, w, h) in rects:
		rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
		# determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		distance = lip_distance(shape)
		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		lip = shape[48:60]
		cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
        
        
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			# if the eyes were closed for a sufficient number of
			# frames, then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if alarm_status == False:
					alarm_status = True
					t = Thread(target=alarm, args=('wake up',))
					t.daemon = True
					t.start()
					#ALARM_ON = True
					# check to see if the TrafficHat buzzer should
					# be sounded
					
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# otherwise, the eye aspect ratio is not below the blink threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			alarm_status = False
		
		if(distance > YAWN_THRESH):
			cv2.putText(frame, "Yawnning", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			if alarm_status2 == False and saying == False:
				alarm_status2 = True
				t = Thread(target=alarm, args=('Take some fresh air',))
				t.daemon = True
				t.start()
		else:
			alarm_status2 = False
			
		cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	# show the frame
	cv2.imshow("Sleep Detection", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
