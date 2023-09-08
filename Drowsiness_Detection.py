
import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imutils
import pygame.mixer as mixer

# Initializing the mixer and load the alert sound
mixer.init()
mixer.music.load("music.wav")

# calculating the eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for detecting drowsiness
thresh = 0.25

# Number of consecutive frames below the threshold to trigger an alert
frame_check = 20

#Initialize face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#Define the indices of the left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

#Open the webcam
cap = cv2.VideoCapture(0)

#Initialize a flag to count frames below the threshold
flag = 0

while True:
    #Read a frame from the webcam
    ret, frame = cap.read()

    #Resize the frame for better processing speed
    frame = imutils.resize(frame, width=450)

    #Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces in the grayscale frame
    subjects = detector(gray, 0)

    for subject in subjects:
        # Predict facial landmarks for the detected face
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the eye aspect ratio (EAR) for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Compute the average EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Checking if the EAR is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)

            # If frames below threshold accumulate, trigger an alert
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # this will play the alert sound
                mixer.music.play()
        else:
            flag = 0

    # Display the processed frame
    cv2.imshow("Drowsy Alert by Karthik", frame)

    # Check for the 'q' key to quit the application
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources and close the webcam
cv2.destroyAllWindows()
cap.release()
