from turtle import width
import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

def midpoint(p1,p2):

    return (int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))


def get_blinking_ratio(eye_points , facial_landmarks):

    left_point = ( facial_landmarks.part(eye_points[0]).x,  facial_landmarks.part(eye_points[0]).y)
    right_point = ( facial_landmarks.part(eye_points[3]).x,  facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    
    #Uncomment this to see the lines
    #hor_line = cv2.line(frame, left_point, right_point,(0,255,0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom,(0,255,0), 2)

    vertical_line_length = hypot(center_top[0]-center_bottom[0], center_top[1]- center_bottom[1])
    horizontal_line_length = hypot(left_point[0]-right_point[0], left_point[1]- right_point[1])

    return horizontal_line_length/vertical_line_length



def get_eye_ratio(eye_points,facial_landmarks):

        #Gaze Detection
#eye_region is a np.array with the coordinates of the eye region for each frame
        left_eye_region = np.array([(landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
            (facial_landmarks.part(eye_points[1]).x,facial_landmarks.part(eye_points[1]).y),
            (facial_landmarks.part(eye_points[2]).x,facial_landmarks.part(eye_points[3]).y),
            (facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y),
            (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
            (facial_landmarks.part(eye_points[5]).x,facial_landmarks.part(eye_points[5]).y)],np.int32)


    #define mask
        height,width,_ = frame.shape
        mask = np.zeros((height, width), np.uint8)
    #only eye
        cv2.polylines(mask, [left_eye_region],True, (0,0,255),2)
        cv2.fillPoly(mask, [left_eye_region],255)
        eye = cv2.bitwise_and(gray, gray, mask = mask)
    #Line that contours the left eye
        #cv2.polylines(frame, [leye_region],True, (0,0,255),2)

        min_x  = np.min(left_eye_region[:,0])
        max_x  = np.max(left_eye_region[:,0])
        min_y  = np.min(left_eye_region[:,1])
        max_y  = np.max(left_eye_region[:,1])

    #Defining only eye

        gray_eye = eye[min_y: max_y,min_x : max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
    
    #Defining eye sides
        left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
    
        right_side_threshold = threshold_eye[0: height, int(width/2): int(width)]
        right_side_white = cv2.countNonZero(right_side_threshold)

    

        ratio = left_side_white/(right_side_white +0.0001)

        return ratio

while True:

    _, frame = cap.read()
    new_frame = np.zeros((500,500,3),np.uint8)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:

        #x,y = face.left(), face.top()
        #x1,y1 = face.right(), face.bottom()

        #cv2.rectangle(frame, (x,y), (x1,y1) , (0,255,0), 2)
    
        landmarks = predictor(gray, face)
#Detect Blink
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        blinking_eye_ratio = (left_eye_ratio + right_eye_ratio)/2
        if (blinking_eye_ratio > 5.7):

            cv2.putText(frame, "BLINKING", (50,150), font, 7, (0,255,0))




        left_eye_ratio = get_eye_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_eye_ratio([42,43,44,45,46,47], landmarks)

        gaze_ratio = (left_eye_ratio+right_eye_ratio)/2
        if (gaze_ratio <= 1):
            cv2.putText(frame, "RIGHT", (50,100), font, 2, (0,2,255),3)
            new_frame[:] = (0,0,255)

        elif (1< gaze_ratio <= 1.7):
            cv2.putText(frame, "CENTER", (50,100), font, 2, (0,2,255),3)
            
        else:

            cv2.putText(frame, "LEFT", (50,100), font, 2, (0,2,255),3)
            new_frame[:] = (255,0,0)
    cv2.imshow ("Frame", frame)
    cv2.imshow ("New Frame", new_frame)

    key = cv2.waitKey(1)

    if key == 27:

        break
cap.release()
cv2.destroyAllWindows()