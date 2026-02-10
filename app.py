import cv2
import mediapipe as mp
import time
import sys
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG & STATE ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 
SCREEN_W, SCREEN_H = pyautogui.size()
S_FACTOR = 0.65 
WINDOW_NAME = 'DeParted'

# Toggle States
show_camera = True
show_face = True
show_hand = True
show_objs = True
mouse_active = True

neutral_tilt = 0.0  
hand_zero_pt = np.array([0.5, 0.5])
current_angle = 0.0
latest_hand, latest_face, latest_objs = None, None, None
smoothed_h = None
clicking = False

# --- SKELETON MAPS ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183]
L_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
HAND_CONN = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17), (17,0)]

# --- UI & LOGIC ---
def handle_click(event, x, y, flags, param):
    global show_camera, show_face, show_hand, show_objs, mouse_active, neutral_tilt, hand_zero_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        # X-coordinate logic for buttons (width ~80px each)
        if 0 <= y <= 40:
            if 0 <= x <= 80: show_camera = not show_camera
            elif 85 <= x <= 165: show_face = not show_face
            elif 170 <= x <= 250: show_hand = not show_hand
            elif 255 <= x <= 335: show_objs = not show_objs
            elif 340 <= x <= 420: mouse_active = not mouse_active
            elif 425 <= x <= 505: 
                neutral_tilt = current_angle
                if smoothed_h is not None: hand_zero_pt = smoothed_h[8]
                print("[SYSTEM] Calibrated.")

def draw_skeleton_loop(img, pts, indices, color):
    for i in range(len(indices) - 1):
        cv2.line(img, tuple(pts[indices[i]]), tuple(pts[indices[i+1]]), color, 1, cv2.LINE_AA)
    cv2.line(img, tuple(pts[indices[-1]]), tuple(pts[indices[0]]), color, 1, cv2.LINE_AA)

def hand_cb(res, img, ts): global latest_hand; latest_hand = res
def face_cb(res, img, ts): global latest_face; latest_face = res
def obj_cb(res, img, ts): global latest_objs; latest_objs = res

# --- INIT ---
def get_user_camera():
    available = []
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened(): available.append(i); cap.release()
    ui = input(f"Select camera {available}: ").strip()
    return int(ui) if ui.isdigit() and int(ui) in available else available[0]

SELECTED_CAM = get_user_camera()
face_tracker_opts = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), running_mode=vision.RunningMode.LIVE_STREAM, result_callback=face_cb)
hand_tracker_opts = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), running_mode=vision.RunningMode.LIVE_STREAM, result_callback=hand_cb)
obj_detector_opts = vision.ObjectDetectorOptions(base_options=python.BaseOptions(model_asset_path='efficientdet_lite0.tflite'), running_mode=vision.RunningMode.LIVE_STREAM, score_threshold=0.5, result_callback=obj_cb)

cap = cv2.VideoCapture(SELECTED_CAM, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, handle_click)

with vision.HandLandmarker.create_from_options(hand_tracker_opts) as hand_tracker, \
     vision.FaceLandmarker.create_from_options(face_tracker_opts) as face_tracker, \
     vision.ObjectDetector.create_from_options(obj_detector_opts) as obj_detector:
    
    prev_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts = int(time.time() * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        hand_tracker.detect_async(mp_img, ts)
        face_tracker.detect_async(mp_img, ts)
        obj_detector.detect_async(mp_img, ts)

        display_frame = frame.copy() if show_camera else np.zeros((h, w, 3), dtype=np.uint8)

        # 1. OBJECTS
        if show_objs and latest_objs and latest_objs.detections:
            for det in latest_objs.detections:
                bbox = det.bounding_box
                cv2.rectangle(display_frame, (int(bbox.origin_x), int(bbox.origin_y)), 
                              (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)), (255, 0, 0), 2)
                cv2.putText(display_frame, det.categories[0].category_name.upper(), (int(bbox.origin_x), int(bbox.origin_y)-10), 0, 0.5, (255, 0, 0), 2)

        # 2. FACE
        if show_face and latest_face and latest_face.face_landmarks:
            f_pts = np.array([[lm.x * w, lm.y * h] for lm in latest_face.face_landmarks[0]]).astype(int)
            for pt in f_pts[::4]: cv2.circle(display_frame, tuple(pt), 1, (0, 255, 0), -1)
            draw_skeleton_loop(display_frame, f_pts, FACE_OVAL, (0, 100, 0))
            draw_skeleton_loop(display_frame, f_pts, LIPS, (255, 255, 255))
            draw_skeleton_loop(display_frame, f_pts, L_EYE, (0, 255, 255))
            draw_skeleton_loop(display_frame, f_pts, R_EYE, (0, 255, 255))
            dx, dy = f_pts[263] - f_pts[33]
            current_angle = np.degrees(np.arctan2(dy, dx))

        # 3. HAND & MOUSE
        if latest_hand and latest_hand.hand_landmarks:
            h_raw = np.array([[lm.x, lm.y] for lm in latest_hand.hand_landmarks[0]])
            smoothed_h = h_raw if smoothed_h is None else (smoothed_h * S_FACTOR) + (h_raw * (1 - S_FACTOR))
            h_pts = (smoothed_h * [w, h]).astype(int)

            if show_hand:
                for s, e in HAND_CONN: cv2.line(display_frame, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 2)
                for pt in h_pts: cv2.circle(display_frame, tuple(pt), 4, (0, 255, 0), -1)

            if mouse_active and smoothed_h[8, 1] < smoothed_h[6, 1] and smoothed_h[12, 1] > smoothed_h[10, 1]:
                offset = (smoothed_h[8] - hand_zero_pt) * 2.5
                mx, my = np.clip(SCREEN_W/2 + (offset[0] * SCREEN_W), 0, SCREEN_W), np.clip(SCREEN_H/2 + (offset[1] * SCREEN_H), 0, SCREEN_H)
                pyautogui.moveTo(mx, my, _pause=False)
            
            pinch = np.linalg.norm(smoothed_h[4] - smoothed_h[8])
            if pinch < 0.04:
                if not clicking: pyautogui.mouseDown(); clicking = True
            elif clicking: pyautogui.mouseUp(); clicking = False

        # UI CONTROL BAR
        def draw_btn(x, label, state):
            color = (0, 150, 0) if state else (0, 0, 150)
            cv2.rectangle(display_frame, (x, 0), (x+80, 40), color, -1)
            cv2.putText(display_frame, label, (x+10, 28), 0, 0.5, (255, 255, 255), 1)

        draw_btn(0, "CAM", show_camera)
        draw_btn(85, "FACE", show_face)
        draw_btn(170, "HAND", show_hand)
        draw_btn(255, "OBJ", show_objs)
        draw_btn(340, "MOUSE", mouse_active)
        cv2.rectangle(display_frame, (425, 0), (505, 40), (150, 75, 0), -1)
        cv2.putText(display_frame, "CALIB", (435, 28), 0, 0.5, (255, 255, 255), 1)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(display_frame, f"FPS: {int(fps)}", (w-100, 40), 0, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()