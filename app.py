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

# Settings as requested
config = {
    "show_camera": False,
    "show_face": True,
    "show_hand": True,
    "show_body": True,
    "show_objs": True,
    "show_joints": True,
    "mouse_active": True,
    "v_cam": False,
    "debug_mode": False
}

# 2026-02-08 Persistent Settings placeholder
persistent = {"Volume": 50, "Repeat": False, "Auto Sleep": False, "Overlay": True}

hand_zero_pt = np.array([0.5, 0.5])
latest_hand, latest_face, latest_objs, latest_pose = None, None, None, None
smoothed_hands = {}
prev_time = time.time()

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
        keys = list(config.keys())
        if idx < len(keys):
            config[keys[idx]] = not config[keys[idx]]
        elif idx == 9 and latest_hand and len(latest_hand.hand_landmarks) > 0:
            hand_zero_pt = np.array([latest_hand.hand_landmarks[0][8].x, latest_hand.hand_landmarks[0][8].y])


def h_cb(res, img, ts): global latest_hand; latest_hand = res


def f_cb(res, img, ts): global latest_face; latest_face = res


def o_cb(res, img, ts): global latest_objs; latest_objs = res


def p_cb(res, img, ts): global latest_pose; latest_pose = res


# --- RUNTIME ---
SEL = 0  # Adjust index if needed
cap = cv2.VideoCapture(SEL)
if platform.system() != "Windows": cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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
        pyvirtualcam.Camera(width=1280, height=720, fps=30,
                            device="/dev/video10" if platform.system() != "Windows" else None) as vcam:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Performance Tracking
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

        # Rendering Skeletons (Always on)
        if config["show_body"] and latest_pose and latest_pose.pose_landmarks:
            p_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_pose.pose_landmarks[0]])
            for s, e in POSE_CONN: cv2.line(display_frame, tuple(p_pts[s]), tuple(p_pts[e]), (0, 0, 255), 2)
            if config["show_joints"]:
                for pt in p_pts: cv2.circle(display_frame, tuple(pt), 3, (255, 255, 255), -1)

        if config["show_face"] and latest_face and latest_face.face_landmarks:
            f_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_face.face_landmarks[0]])
            cv2.polylines(display_frame, [np.array([f_pts[i] for i in FACE_OVAL], np.int32)], True, (0, 255, 0), 1)
            if config["show_joints"]:
                for lm in latest_face.face_landmarks[0]: cv2.circle(display_frame, (int(lm.x * w), int(lm.y * h)), 1,
                                                                    (200, 255, 200), -1)

        if config["show_hand"] and latest_hand:
            for idx, hand_lms in enumerate(latest_hand.hand_landmarks):
                h_raw = np.array([[lm.x, lm.y] for lm in hand_lms])
                if idx not in smoothed_hands:
                    smoothed_hands[idx] = h_raw
                else:
                    smoothed_hands[idx] = (smoothed_hands[idx] * 0.65) + (h_raw * 0.35)
                h_pts = (smoothed_hands[idx] * [w, h]).astype(int)
                for s, e in HAND_CONN: cv2.line(display_frame, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 1)
                if config["show_joints"]:
                    for pt in h_pts: cv2.circle(display_frame, tuple(pt), 3, (0, 255, 255), -1)
                if config["mouse_active"] and idx == 0:
                    off = (smoothed_hands[0][8] - hand_zero_pt) * 2.5
                    pyautogui.moveTo(np.clip(SCREEN_W / 2 + (off[0] * SCREEN_W), 0, SCREEN_W),
                                     np.clip(SCREEN_H / 2 + (off[1] * SCREEN_H), 0, SCREEN_H), _pause=False)

        if config["show_objs"] and latest_objs:
            for d in latest_objs.detections:
                b = d.bounding_box
                cv2.rectangle(display_frame, (int(b.origin_x), int(b.origin_y)),
                              (int(b.origin_x + b.width), int(b.origin_y + b.height)), (255, 0, 0), 2)

        # Debug Stats
        if config["debug_mode"]:
            stats = [f"FPS: {int(fps)}", f"CPU: {psutil.cpu_percent()}%", f"RAM: {psutil.virtual_memory().percent}%"]
            for i, text in enumerate(stats):
                cv2.putText(display_frame, text, (10, h - 20 - (i * 30)), 0, 0.7, (0, 255, 255), 2)

        # Settings Menu Bar
        labels = ["CAM", "FACE", "HAND", "BODY", "OBJ", "JOINT", "MSE", "VCAM", "DEBUG", "CALIB"]
        for i, label in enumerate(labels):
            key = list(config.keys())[i] if i < 9 else None
            state = config[key] if key else True
            cv2.rectangle(display_frame, (i * 80, 0), (i * 80 + 75, 40), (0, 150, 0) if state else (0, 0, 150), -1)
            cv2.putText(display_frame, label, (i * 80 + 5, 25), 0, 0.35, (255, 255, 255), 1)

        # Virtual Cam Output
        if config["v_cam"]:
            vcam.send(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            vcam.sleep_until_next_frame()

        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()