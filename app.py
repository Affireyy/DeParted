import cv2
import mediapipe as mp
import time
import sys
import os
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG & STATE ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 
SCREEN_W, SCREEN_H = pyautogui.size()
S_FACTOR = 0.65 
WINDOW_NAME = 'Holistic Pro Full Command'

# Toggle States
show_camera, show_face, show_hand, show_body, show_objs, mouse_active = True, True, True, True, True, True
neutral_tilt, hand_zero_pt, current_angle = 0.0, np.array([0.5, 0.5]), 0.0
latest_hand, latest_face, latest_objs, latest_pose = None, None, None, None
smoothed_h = None
clicking = False

# --- SKELETON MAPS ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
L_IRIS = [474, 475, 476, 477]
R_IRIS = [469, 470, 471, 472]
HAND_CONN = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17), (17,0)]
POSE_CONN = [(11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (24,26), (25,27), (26,28), (27,29), (28,30), (29,31), (30,32)]

# --- CAMERA SELECTOR ---
def get_user_camera():
    available = []
    print("\n[SYSTEM] Scanning for cameras...")
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: available.append(i)
            cap.release()
    if not available: sys.exit("!!! No cameras detected.")
    print(f"Detected Camera Indices: {available}")
    user_input = input(f"Select index (Default {available[0]}): ").strip()
    try:
        if user_input == "": return available[0]
        choice = int(user_input)
        return choice if choice in available else available[0]
    except ValueError:
        return available[0]

# --- UI HANDLER ---
def handle_click(event, x, y, flags, param):
    global show_camera, show_face, show_hand, show_body, show_objs, mouse_active, neutral_tilt, hand_zero_pt
    if event == cv2.EVENT_LBUTTONDOWN and 0 <= y <= 40:
        idx = x // 80
        if idx == 0: show_camera = not show_camera
        elif idx == 1: show_face = not show_face
        elif idx == 2: show_hand = not show_hand
        elif idx == 3: show_body = not show_body
        elif idx == 4: show_objs = not show_objs
        elif idx == 5: mouse_active = not mouse_active
        elif idx == 6: 
            neutral_tilt = current_angle
            if smoothed_h is not None: hand_zero_pt = smoothed_h[8]

# --- CALLBACKS ---
def h_cb(res, img, ts): global latest_hand; latest_hand = res
def f_cb(res, img, ts): global latest_face; latest_face = res
def o_cb(res, img, ts): global latest_objs; latest_objs = res
def p_cb(res, img, ts): global latest_pose; latest_pose = res

# --- MAIN ---
SELECTED_CAM = get_user_camera()
cap = cv2.VideoCapture(SELECTED_CAM, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, handle_click)

with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), running_mode=vision.RunningMode.LIVE_STREAM, result_callback=h_cb)) as hand_tk, \
     vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), running_mode=vision.RunningMode.LIVE_STREAM, result_callback=f_cb)) as face_tk, \
     vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(base_options=python.BaseOptions(model_asset_path='efficientdet_lite0.tflite'), running_mode=vision.RunningMode.LIVE_STREAM, score_threshold=0.5, result_callback=o_cb)) as obj_tk, \
     vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='pose_landmarker.task'), running_mode=vision.RunningMode.LIVE_STREAM, result_callback=p_cb)) as pose_tk:

    prev_time = time.time()
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1); h, w = frame.shape[:2]; ts = int(time.time() * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        hand_tk.detect_async(mp_img, ts); face_tk.detect_async(mp_img, ts)
        obj_tk.detect_async(mp_img, ts); pose_tk.detect_async(mp_img, ts)

        display_frame = frame.copy() if show_camera else np.zeros((h, w, 3), dtype=np.uint8)

        # 1. BODY JOINTS
        if show_body and latest_pose and len(latest_pose.pose_landmarks) > 0:
            p_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_pose.pose_landmarks[0]])
            for s, e in POSE_CONN: cv2.line(display_frame, tuple(p_pts[s]), tuple(p_pts[e]), (0, 0, 255), 2)
            for pt in p_pts: cv2.circle(display_frame, tuple(pt), 3, (255, 255, 255), -1)

        # 2. FACE JOINTS (Eyes, Lips, Irises)
        if show_face and latest_face and len(latest_face.face_landmarks) > 0:
            f_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_face.face_landmarks[0]])
            for loop in [FACE_OVAL, L_EYE, R_EYE, LIPS]:
                pts_loop = np.array([f_pts[idx] for idx in loop], np.int32)
                cv2.polylines(display_frame, [pts_loop], True, (0, 255, 0), 1)
            for iris in [L_IRIS, R_IRIS]:
                center = np.mean([f_pts[idx] for idx in iris], axis=0).astype(int)
                cv2.circle(display_frame, tuple(center), 2, (0, 0, 255), -1)

        # 3. HAND JOINTS & MOUSE
        if latest_hand and len(latest_hand.hand_landmarks) > 0:
            h_raw = np.array([[lm.x, lm.y] for lm in latest_hand.hand_landmarks[0]])
            smoothed_h = h_raw if smoothed_h is None else (smoothed_h * S_FACTOR) + (h_raw * (1 - S_FACTOR))
            h_pts = (smoothed_h * [w, h]).astype(int)
            if show_hand:
                for s, e in HAND_CONN: cv2.line(display_frame, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 1)
                for pt in h_pts: cv2.circle(display_frame, tuple(pt), 3, (0, 255, 255), -1)
            if mouse_active:
                offset = (smoothed_h[8] - hand_zero_pt) * 2.5
                pyautogui.moveTo(np.clip(SCREEN_W/2 + (offset[0] * SCREEN_W), 0, SCREEN_W), np.clip(SCREEN_H/2 + (offset[1] * SCREEN_H), 0, SCREEN_H), _pause=False)

        # 4. OBJECTS WITH LABELS
        if show_objs and latest_objs:
            for d in latest_objs.detections:
                b = d.bounding_box
                label = d.categories[0].category_name
                cv2.rectangle(display_frame, (int(b.origin_x), int(b.origin_y)), (int(b.origin_x+b.width), int(b.origin_y+b.height)), (255, 0, 0), 2)
                cv2.putText(display_frame, label.upper(), (int(b.origin_x), int(b.origin_y)-10), 0, 0.5, (255, 0, 0), 2)

        # UI Bar
        btns = [("CAM", show_camera), ("FACE", show_face), ("HAND", show_hand), ("BODY", show_body), ("OBJ", show_objs), ("MOUSE", mouse_active), ("CALIB", True)]
        for i, (label, state) in enumerate(btns):
            col = (0, 150, 0) if state else (0, 0, 150)
            if label == "CALIB": col = (150, 75, 0)
            cv2.rectangle(display_frame, (i*80, 0), (i*80+75, 40), col, -1)
            cv2.putText(display_frame, label, (i*80+5, 25), 0, 0.4, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()