import threading
import time
import cv2
import imutils
import pygame
import sys
import os
import numpy as np
import yaml

pygame.mixer.init()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

classes = open('models/yolo/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv2.dnn.readNetFromDarknet('models/yolo/yolov3.cfg', 'models/yolo/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

droidcam_url = config['camera_url']
cap = cv2.VideoCapture(droidcam_url)

desired_width = config['frame_width']
desired_height = config['frame_height']
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
cap.set(cv2.CAP_PROP_FPS, config['fps'])

time.sleep(2)

ret, start_frame = cap.read()
if not ret:
    print("Failed to capture initial frame from camera.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

start_frame = imutils.resize(start_frame, width=desired_width)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

alarm = False
alarm_mode = False
alarm_counter = 0
full_screen = False
frame_skip = 2
frame_count = 0
music_playing = False

fps_start_time = time.time()
fps = 0

last_detection_time = {}

def play_music():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        pygame.mixer.music.load(config['alarm_sound'])
        pygame.mixer.music.play()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    frame = imutils.resize(frame, width=desired_width)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    frame_delta = cv2.absdiff(start_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        rois.append((x, y, w, h))
    
    if alarm_mode:
        frame_count += 1
        if frame_count % frame_skip == 0:
            for (x, y, w, h) in rois:
                roi = frame[y:y+h, x:x+w]
                blob = cv2.dnn.blobFromImage(roi, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                t0 = time.time()
                outputs = net.forward(ln)
                t = time.time()
                print(f'YOLO detection time: {t-t0}, FPS: {1/(t-t0):.2f}, Output: {len(outputs)}')

                boxes = []
                confidences = []
                classIDs = []
                roi_h, roi_w = roi.shape[:2]

                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if confidence > 0.5:
                            box = detection[:4] * np.array([roi_w, roi_h, roi_w, roi_h])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            box = [x, y, int(width), int(height)]
                            boxes.append(box)
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                if len(indices) > 0:
                    for i in indices.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        color = [int(c) for c in colors[classIDs[i]]]
                        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                        cv2.putText(roi, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        print(f"Detected {classes[classIDs[i]]} with confidence {confidences[i]:.4f} at [{x}, {y}, {w}, {h}]")
                        
                        focal_length = 615
                        real_height = 1.75
                        distance = (real_height * focal_length) / h
                        distance_text = f"Distance: {distance:.2f}m"
                        cv2.putText(roi, distance_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        last_detection_time[i] = time.time()
                        
                        if classes[classIDs[i]] == "person" and not music_playing:
                            music_playing = True
                            threading.Thread(target=play_music).start()

    current_time = time.time()
    for i in list(last_detection_time.keys()):
        if current_time - last_detection_time[i] < 10:
            if i < len(boxes):
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                distance_text = f"Distance: {distance:.2f}m"
                cv2.putText(frame, distance_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            del last_detection_time[i]

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    if time_diff > 0:
        fps = 1 / time_diff
    else:
        fps = 0
    fps_start_time = fps_end_time
    fps_text = f"{int(fps)}"
    text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    cv2.imshow("Cam", frame)
        
    key_pressed = cv2.waitKey(30)
    if key_pressed == ord("t"):
        alarm_mode = not alarm_mode
        alarm_counter = 0
    elif key_pressed == ord("g"):
        pygame.mixer.music.stop()
        music_playing = False
    elif key_pressed == 0:
        full_screen = not full_screen
        if full_screen:
            cv2.setWindowProperty("Cam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Cam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key_pressed == ord("q"):
        alarm_mode = False
        break
    
cap.release()
cv2.destroyAllWindows()