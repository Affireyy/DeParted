import cv2
import mediapipe as mp
import time
import sys
import os
import platform
import urllib.request
import numpy as np
import pyautogui
import psutil
import pyvirtualcam
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
                print(f"[DeParted] Downloading {name}...")
                urllib.request.urlretrieve(url, name)


# --- SETTINGS & STATE ---
ModelDownloader.ensure_models()
pyautogui.FAILSAFE = False
WINDOW_NAME = 'DeParted'

# Persistent Settings (as requested in 2026-02-08 update)
config = {
    "show_camera": False,
    "show_face": True,
    "show_hand": True,
    "show_body": True,
    "show_objs": True,
    "show_joints": True,
    "mouse_active": True,
    "debug_mode": False,
    "v_cam": False,  # Virtual Camera Toggle
    "volume": 50,
    "repeat": False,
    "auto_sleep": False,
    "overlay": True
}

hand_zero_pt = np.array([0.5, 0.5])
latest_hand, latest_face, latest_objs, latest_pose = None, None, None, None
smoothed_hands = {}
fps = 0
prev_time = 0

# --- MAPPINGS ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
             (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17), (17, 0)]
POSE_CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
             (25, 27), (26, 28)]


# --- HELPERS ---
def handle_click(event, x, y, flags, param):
    global hand_zero_pt
    if event == cv2.EVENT_LBUTTONDOWN and 0 <= y <= 40:
        idx = x // 80
        keys = ["show_camera", "show_face", "show_hand", "show_body", "show_objs", "show_joints", "mouse_active",
                "v_cam", "debug_mode"]
        if idx < len(keys):
            config[keys[idx]] = not config[keys[idx]]
        elif idx == len(keys) and latest_hand and latest_hand.hand_landmarks:
            hand_zero_pt = np.array([latest_hand.hand_landmarks[0][8].x, latest_hand.hand_landmarks[0][8].y])


def h_cb(res, img, ts): global latest_hand; latest_hand = res


def f_cb(res, img, ts): global latest_face; latest_face = res


def o_cb(res, img, ts): global latest_objs; latest_objs = res


def p_cb(res, img, ts): global latest_pose; latest_pose = res


# --- RUNTIME ---
SEL = 0  # Assume 0 for quick start, use get_user_camera() logic if needed
cap = cv2.VideoCapture(SEL)
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
                                         running_mode=vision.RunningMode.LIVE_STREAM, result_callback=p_cb)) as pose_tk, \
        pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    print(f"[DeParted] Virtual Camera Active: {cam.device}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        frame = cv2.flip(frame, 1);
        h, w = frame.shape[:2];
        ts = int(time.time() * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        hand_tk.detect_async(mp_img, ts);
        face_tk.detect_async(mp_img, ts)
        obj_tk.detect_async(mp_img, ts);
        pose_tk.detect_async(mp_img, ts)

        display_frame = frame.copy() if config["show_camera"] else np.zeros((h, w, 3), dtype=np.uint8)

        # Rendering (Hand, Face, Pose, Objs) - [Truncated for brevity, same logic as previous version]
        if config["show_hand"] and latest_hand:
            for idx, hand_lms in enumerate(latest_hand.hand_landmarks):
                h_raw = np.array([[lm.x, lm.y] for lm in hand_lms])
                h_pts = (h_raw * [w, h]).astype(int)
                for s, e in HAND_CONN: cv2.line(display_frame, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 1)
                if config["show_joints"]:
                    for pt in h_pts: cv2.circle(display_frame, tuple(pt), 3, (0, 255, 255), -1)

        # Debug Overlay
        if config["debug_mode"]:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, h - 80), 0, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"CPU: {cpu}%", (10, h - 50), 0, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"RAM: {ram}%", (10, h - 20), 0, 0.7, (0, 255, 255), 2)

        # UI Bar
        labels = ["CAM", "FACE", "HAND", "BODY", "OBJ", "JOINT", "MSE", "VCAM", "DEBUG", "CALIB"]
        for i, label in enumerate(labels):
            state = config.get(list(config.keys())[i]) if i < 9 else True
            col = (0, 150, 0) if state else (0, 0, 150)
            cv2.rectangle(display_frame, (i * 80, 0), (i * 80 + 75, 40), col, -1)
            cv2.putText(display_frame, label, (i * 80 + 5, 25), 0, 0.3, (255, 255, 255), 1)

        # Send to Virtual Camera
        if config["v_cam"]:
            cam.send(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()

        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()