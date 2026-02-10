import cv2
import mediapipe as mp
import time
import sys
import os
import platform
import urllib.request
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- AUTO-DOWNLOADER ---
class ModelDownloader:
    MODELS = {
        'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        'efficientdet_lite0.tflite': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite'
    }

    @staticmethod
    def ensure_models():
        for name, url in ModelDownloader.MODELS.items():
            if not os.path.exists(name):
                print(f"[DeParted] Downloading missing model: {name}...")
                urllib.request.urlretrieve(url, name)


# --- CONFIG & STATE ---
ModelDownloader.ensure_models()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()
S_FACTOR = 0.65
WINDOW_NAME = 'DeParted - Holistic Tracking'

# Defaults: CAM is off, JOINTS is a toggle, Skeletons always show.
show_camera = False
show_face, show_hand, show_body, show_objs, show_joints, mouse_active = True, True, True, True, True, True

hand_zero_pt = np.array([0.5, 0.5])
latest_hand, latest_face, latest_objs, latest_pose = None, None, None, None
smoothed_hands = {}  # Track smoothing for multiple hands

# --- LANDMARK MAPPINGS ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 291]
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
             (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17), (17, 0)]
POSE_CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
             (25, 27), (26, 28)]


# --- HELPERS ---
def get_user_camera():
    available = []
    os_type = platform.system()
    for i in range(5):
        backend = cv2.CAP_DSHOW if os_type == "Windows" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened(): cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read();
            available.append(i) if ret else None
            cap.release()
    if not available: sys.exit("No cameras found.")
    print(f"DeParted detected cameras: {available}")
    u = input(f"Select index [{available[0]}]: ").strip()
    return int(u) if u.isdigit() and int(u) in available else available[0]


def handle_click(event, x, y, flags, param):
    global show_camera, show_face, show_hand, show_body, show_objs, show_joints, mouse_active, hand_zero_pt
    if event == cv2.EVENT_LBUTTONDOWN and 0 <= y <= 40:
        idx = x // 80
        if idx == 0:
            show_camera = not show_camera
        elif idx == 1:
            show_face = not show_face
        elif idx == 2:
            show_hand = not show_hand
        elif idx == 3:
            show_body = not show_body
        elif idx == 4:
            show_objs = not show_objs
        elif idx == 5:
            show_joints = not show_joints
        elif idx == 6:
            mouse_active = not mouse_active
        elif idx == 7 and latest_hand and len(latest_hand.hand_landmarks) > 0:
            hand_zero_pt = np.array([latest_hand.hand_landmarks[0][8].x, latest_hand.hand_landmarks[0][8].y])


def h_cb(res, img, ts): global latest_hand; latest_hand = res


def f_cb(res, img, ts): global latest_face; latest_face = res


def o_cb(res, img, ts): global latest_objs; latest_objs = res


def p_cb(res, img, ts): global latest_pose; latest_pose = res


# --- RUNTIME ---
SEL = get_user_camera()
OS = platform.system()
cap = cv2.VideoCapture(SEL, cv2.CAP_DSHOW if OS == "Windows" else cv2.CAP_V4L2)
if OS != "Windows": cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, handle_click)

with vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
                                     running_mode=vision.RunningMode.LIVE_STREAM, num_hands=2,
                                     result_callback=h_cb)) as hand_tk, \
        vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
                                         running_mode=vision.RunningMode.LIVE_STREAM, result_callback=f_cb)) as face_tk, \
        vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(base_options=python.BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
                                         running_mode=vision.RunningMode.LIVE_STREAM, score_threshold=0.5,
                                         result_callback=o_cb)) as obj_tk, \
        vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='pose_landmarker.task'),
                                         running_mode=vision.RunningMode.LIVE_STREAM, result_callback=p_cb)) as pose_tk:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1);
        h, w = frame.shape[:2];
        ts = int(time.time() * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        hand_tk.detect_async(mp_img, ts);
        face_tk.detect_async(mp_img, ts)
        obj_tk.detect_async(mp_img, ts);
        pose_tk.detect_async(mp_img, ts)

        display_frame = frame.copy() if show_camera else np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Pose Skeleton
        if show_body and latest_pose and latest_pose.pose_landmarks:
            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_pose.pose_landmarks[0]])
            for s, e in POSE_CONN: cv2.line(display_frame, tuple(pts[s]), tuple(pts[e]), (0, 0, 255), 2)
            if show_joints:
                for pt in pts: cv2.circle(display_frame, tuple(pt), 3, (255, 255, 255), -1)

        # 2. Face Skeleton
        if show_face and latest_face and latest_face.face_landmarks:
            f_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_face.face_landmarks[0]])
            for loop in [FACE_OVAL, L_EYE, R_EYE, LIPS]:
                cv2.polylines(display_frame, [np.array([f_pts[i] for i in loop], np.int32)], True, (0, 255, 0), 1)
            if show_joints:
                for lm in latest_face.face_landmarks[0]: cv2.circle(display_frame, (int(lm.x * w), int(lm.y * h)), 1,
                                                                    (200, 255, 200), -1)

        # 3. Multi-Hand Skeleton
        if show_hand and latest_hand and latest_hand.hand_landmarks:
            for idx, hand_lms in enumerate(latest_hand.hand_landmarks):
                h_raw = np.array([[lm.x, lm.y] for lm in hand_lms])
                if idx not in smoothed_hands:
                    smoothed_hands[idx] = h_raw
                else:
                    smoothed_hands[idx] = (smoothed_hands[idx] * S_FACTOR) + (h_raw * (1 - S_FACTOR))

                h_pts = (smoothed_hands[idx] * [w, h]).astype(int)
                for s, e in HAND_CONN: cv2.line(display_frame, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 1)
                if show_joints:
                    for pt in h_pts: cv2.circle(display_frame, tuple(pt), 3, (0, 255, 255), -1)

                # Mouse control on primary hand
                if mouse_active and idx == 0:
                    off = (smoothed_hands[0][8] - hand_zero_pt) * 2.5
                    pyautogui.moveTo(np.clip(SCREEN_W / 2 + (off[0] * SCREEN_W), 0, SCREEN_W),
                                     np.clip(SCREEN_H / 2 + (off[1] * SCREEN_H), 0, SCREEN_H), _pause=False)

        # 4. Objects
        if show_objs and latest_objs:
            for d in latest_objs.detections:
                b = d.bounding_box
                cv2.rectangle(display_frame, (int(b.origin_x), int(b.origin_y)),
                              (int(b.origin_x + b.width), int(b.origin_y + b.height)), (255, 0, 0), 2)

        # UI Bar
        btns = [("CAM", show_camera), ("FACE", show_face), ("HAND", show_hand), ("BODY", show_body), ("OBJ", show_objs),
                ("JOINTS", show_joints), ("MOUSE", mouse_active), ("CALIB", True)]
        for i, (label, state) in enumerate(btns):
            col = (0, 150, 0) if state else (0, 0, 150)
            cv2.rectangle(display_frame, (i * 80, 0), (i * 80 + 75, 40), col, -1)
            cv2.putText(display_frame, label, (i * 80 + 5, 25), 0, 0.35, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()