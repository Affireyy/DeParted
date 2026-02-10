import cv2
import mediapipe as mp
import time
import os
import platform
import urllib.request
import numpy as np
import pyautogui
import psutil
import pyvirtualcam
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- INITIALIZATION ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()
WINDOW_MAIN = 'DeParted - Stream Output'
WINDOW_SET = 'DeParted - Settings'

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


# --- CALLBACKS ---
def handle_settings_click(event, x, y, flags, param):
    global hand_zero_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = y // 50  # Vertical menu
        keys = list(config.keys())
        if idx < len(keys):
            config[keys[idx]] = not config[keys[idx]]
        elif idx == len(keys) and latest_hand and len(latest_hand.hand_landmarks) > 0:
            hand_zero_pt = np.array([latest_hand.hand_landmarks[0][8].x, latest_hand.hand_landmarks[0][8].y])


def h_cb(res, img, ts): global latest_hand; latest_hand = res


def f_cb(res, img, ts): global latest_face; latest_face = res


def o_cb(res, img, ts): global latest_objs; latest_objs = res


def p_cb(res, img, ts): global latest_pose; latest_pose = res


# --- RUNTIME ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
cv2.namedWindow(WINDOW_SET, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_SET, 250, 550)
cv2.setMouseCallback(WINDOW_SET, handle_settings_click)

vcam_device = "/dev/video10" if platform.system() != "Windows" else None

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
        pyvirtualcam.Camera(width=1280, height=720, fps=30, device=vcam_device) as vcam:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        curr_time = time.time();
        fps = 1 / (curr_time - prev_time);
        prev_time = curr_time
        frame = cv2.flip(frame, 1);
        h, w = frame.shape[:2];
        ts = int(time.time() * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for tk in [hand_tk, face_tk, obj_tk, pose_tk]: tk.detect_async(mp_img, ts)

        # BUILD CLEAN STREAM FRAME
        stream_frame = frame.copy() if config["show_camera"] else np.zeros((h, w, 3), dtype=np.uint8)

        if config["show_body"] and latest_pose and latest_pose.pose_landmarks:
            p_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_pose.pose_landmarks[0]])
            for s, e in POSE_CONN: cv2.line(stream_frame, tuple(p_pts[s]), tuple(p_pts[e]), (0, 0, 255), 2)
            if config["show_joints"]:
                for pt in p_pts: cv2.circle(stream_frame, tuple(pt), 3, (255, 255, 255), -1)

        if config["show_face"] and latest_face and latest_face.face_landmarks:
            f_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_face.face_landmarks[0]])
            cv2.polylines(stream_frame, [np.array([f_pts[i] for i in FACE_OVAL], np.int32)], True, (0, 255, 0), 1)

        if config["show_hand"] and latest_hand:
            for idx, hand_lms in enumerate(latest_hand.hand_landmarks):
                h_raw = np.array([[lm.x, lm.y] for lm in hand_lms])
                if idx not in smoothed_hands:
                    smoothed_hands[idx] = h_raw
                else:
                    smoothed_hands[idx] = (smoothed_hands[idx] * 0.7) + (h_raw * 0.3)
                h_pts = (smoothed_hands[idx] * [w, h]).astype(int)
                for s, e in HAND_CONN: cv2.line(stream_frame, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 1)
                if config["show_joints"]:
                    for pt in h_pts: cv2.circle(stream_frame, tuple(pt), 3, (0, 255, 255), -1)
                if config["mouse_active"] and idx == 0:
                    off = (smoothed_hands[0][8] - hand_zero_pt) * 2.5
                    pyautogui.moveTo(np.clip(SCREEN_W / 2 + (off[0] * SCREEN_W), 0, SCREEN_W),
                                     np.clip(SCREEN_H / 2 + (off[1] * SCREEN_H), 0, SCREEN_H), _pause=False)

        if config["debug_mode"]:
            stats = [f"FPS: {int(fps)}", f"CPU: {psutil.cpu_percent()}%", f"RAM: {psutil.virtual_memory().percent}%"]
            for i, txt in enumerate(stats): cv2.putText(stream_frame, txt, (10, h - 20 - (i * 25)), 0, 0.6,
                                                        (0, 255, 255), 2)

        # BUILD SETTINGS WINDOW
        settings_panel = np.zeros((550, 250, 3), dtype=np.uint8)
        for i, (key, val) in enumerate(config.items()):
            color = (0, 200, 0) if val else (0, 0, 200)
            cv2.rectangle(settings_panel, (10, i * 50 + 5), (240, i * 50 + 45), color, -1)
            cv2.putText(settings_panel, f"{key.replace('show_', '').upper()}: {val}", (20, i * 50 + 30), 0, 0.5,
                        (255, 255, 255), 1)

        # BLUE CALIBRATION BUTTON (#0000FF is BGR 255,0,0)
        cv2.rectangle(settings_panel, (10, 455), (240, 495), (255, 0, 0), -1)
        cv2.putText(settings_panel, "CALIBRATE MOUSE", (45, 480), 0, 0.5, (255, 255, 255), 1)

        # OUTPUT
        if config["v_cam"]:
            vcam.send(cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB))
            vcam.sleep_until_next_frame()

        cv2.imshow(WINDOW_MAIN, stream_frame)
        cv2.imshow(WINDOW_SET, settings_panel)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()