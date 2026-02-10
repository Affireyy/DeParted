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
WINDOW_MAIN = 'DeParted'
TARGET_W, TARGET_H = 1280, 720

config = {
    "show_camera": False,
    "show_face": True,
    "show_hand": True,
    "show_body": True,
    "show_objs": True,
    "show_joints": True,
    "mouse_active": True,
    "v_cam": False,
    "debug_mode": False,
    "mouth_click": False
}

hand_zero_pt = np.array([0.5, 0.5])
latest_hand, latest_face, latest_objs, latest_pose = None, None, None, None
smoothed_hands = {}
prev_time = time.time()
mouth_is_open = False  # Track state for clicking

# --- DETAILED FACE MAPPINGS ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
R_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
             (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17), (17, 0)]
POSE_CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
             (25, 27), (26, 28)]


# --- CALLBACKS ---
def handle_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and 0 <= y <= 40:
        idx = x // 80
        keys = list(config.keys())
        if idx < len(keys):
            config[keys[idx]] = not config[keys[idx]]


def h_cb(res, img, ts): global latest_hand; latest_hand = res


def f_cb(res, img, ts): global latest_face; latest_face = res


def o_cb(res, img, ts): global latest_objs; latest_objs = res


def p_cb(res, img, ts): global latest_pose; latest_pose = res


# --- RUNTIME ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_MAIN, TARGET_W, TARGET_H)
cv2.setMouseCallback(WINDOW_MAIN, handle_click)

vcam_device = "/dev/video10" if platform.system() != "Windows" else None

try:
    with vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
                                         running_mode=vision.RunningMode.LIVE_STREAM, num_hands=2,
                                         result_callback=h_cb)) as hand_tk, \
            vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
                                             running_mode=vision.RunningMode.LIVE_STREAM,
                                             result_callback=f_cb)) as face_tk, \
            vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
                running_mode=vision.RunningMode.LIVE_STREAM, score_threshold=0.5, result_callback=o_cb)) as obj_tk, \
            vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='pose_landmarker.task'),
                                             running_mode=vision.RunningMode.LIVE_STREAM,
                                             result_callback=p_cb)) as pose_tk, \
            pyvirtualcam.Camera(width=TARGET_W, height=TARGET_H, fps=30, device=vcam_device) as vcam:

        while cap.isOpened():
            success, raw_frame = cap.read()
            if not success: break

            frame = cv2.resize(raw_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            curr_time = time.time();
            fps_val = 1 / (curr_time - prev_time);
            prev_time = curr_time
            frame = cv2.flip(frame, 1);
            h, w = frame.shape[:2];
            ts = int(time.time() * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            for tk in [hand_tk, face_tk, obj_tk, pose_tk]: tk.detect_async(mp_img, ts)

            output_frame = frame.copy() if config["show_camera"] else np.zeros((h, w, 3), dtype=np.uint8)

            # 1. Pose
            if config["show_body"] and latest_pose and latest_pose.pose_landmarks:
                p_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_pose.pose_landmarks[0]])
                for s, e in POSE_CONN: cv2.line(output_frame, tuple(p_pts[s]), tuple(p_pts[e]), (0, 0, 255), 2)
                if config["show_joints"]:
                    for pt in p_pts: cv2.circle(output_frame, tuple(pt), 3, (255, 255, 255), -1)

            # 2. Face
            if config["show_face"] and latest_face and latest_face.face_landmarks:
                # FIXED: Declare global here at the start of the block
                global mouth_is_open
                f_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in latest_face.face_landmarks[0]])

                for loop in [FACE_OVAL, L_EYE, R_EYE, L_BROW, R_BROW, MOUTH_OUTER, MOUTH_INNER]:
                    cv2.polylines(output_frame, [np.array([f_pts[i] for i in loop], np.int32)], True, (0, 255, 0), 1)

                if config["show_joints"]:
                    for lm in latest_face.face_landmarks[0]: cv2.circle(output_frame, (int(lm.x * w), int(lm.y * h)), 1,
                                                                        (200, 255, 200), -1)

                # Mouth Click
                if config["mouse_active"] and config["mouth_click"]:
                    # Distance between landmark 13 (top lip) and 14 (bottom lip)
                    dist = np.linalg.norm(
                        np.array([latest_face.face_landmarks[0][13].x, latest_face.face_landmarks[0][13].y]) -
                        np.array([latest_face.face_landmarks[0][14].x, latest_face.face_landmarks[0][14].y]))
                    if dist > 0.04:
                        if not mouth_is_open:
                            pyautogui.click()
                            mouth_is_open = True
                    else:
                        mouth_is_open = False

            # 3. Hands
            if config["show_hand"] and latest_hand:
                for idx, hand_lms in enumerate(latest_hand.hand_landmarks):
                    h_raw = np.array([[lm.x, lm.y] for lm in hand_lms])
                    if idx not in smoothed_hands:
                        smoothed_hands[idx] = h_raw
                    else:
                        smoothed_hands[idx] = (smoothed_hands[idx] * 0.7) + (h_raw * 0.3)
                    h_pts = (smoothed_hands[idx] * [w, h]).astype(int)
                    for s, e in HAND_CONN: cv2.line(output_frame, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 1)
                    if config["show_joints"]:
                        for pt in h_pts: cv2.circle(output_frame, tuple(pt), 3, (0, 255, 255), -1)
                    if config["mouse_active"] and idx == 0:
                        off = (smoothed_hands[0][8] - hand_zero_pt) * 2.5
                        pyautogui.moveTo(np.clip(SCREEN_W / 2 + (off[0] * SCREEN_W), 0, SCREEN_W),
                                         np.clip(SCREEN_H / 2 + (off[1] * SCREEN_H), 0, SCREEN_H), _pause=False)

            if config["show_objs"] and latest_objs:
                for d in latest_objs.detections:
                    b = d.bounding_box
                    cv2.rectangle(output_frame, (int(b.origin_x), int(b.origin_y)),
                                  (int(b.origin_x + b.width), int(b.origin_y + b.height)), (255, 0, 0), 2)

            if config["v_cam"]:
                vcam.send(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
                vcam.sleep_until_next_frame()

            display_frame = output_frame.copy()
            if config["debug_mode"]:
                stats = [f"FPS: {int(fps_val)}", f"CPU: {psutil.cpu_percent()}%",
                         f"RAM: {psutil.virtual_memory().percent}%"]
                for i, txt in enumerate(stats): cv2.putText(display_frame, txt, (10, h - 20 - (i * 25)), 0, 0.6,
                                                            (0, 255, 255), 2)

            labels = ["CAM", "FACE", "HAND", "BODY", "OBJ", "JOINT", "MSE", "VCAM", "DEBUG", "M-CLK"]
            for i, label in enumerate(labels):
                key = list(config.keys())[i]
                col = (0, 150, 0) if config[key] else (0, 0, 150)
                cv2.rectangle(display_frame, (i * 80, 0), (i * 80 + 75, 40), col, -1)
                cv2.putText(display_frame, label, (i * 80 + 5, 25), 0, 0.3, (255, 255, 255), 1)

            cv2.imshow(WINDOW_MAIN, display_frame)
            if cv2.waitKey(1) & 0xFF == 27: break
except KeyboardInterrupt:
    print("[DeParted] Closing safely...")

cap.release()
cv2.destroyAllWindows()