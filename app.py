import cv2
import mediapipe as mp
import time
import os
import platform
import numpy as np
import pyautogui
import psutil
import pyvirtualcam
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- WINDOWS VOLUME INITIALIZATION ---
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

volume_ctrl = None
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    device_enumerator = AudioUtilities.GetDeviceEnumerator()
    default_speakers = device_enumerator.GetDefaultAudioEndpoint(0, 0)
    interface = default_speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
except Exception as e:
    print(f"Volume Init Error: {e}")

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
    "debug_mode": True,
    "mouth_click": True,
    "vol_gest": True,
    "obj_thresh": 0.5
}

state = {
    "hand_zero_pt": np.array([0.5, 0.5]),
    "mouth_was_open": False,
    "prev_time": time.time(),
    "latest_hand": None,
    "latest_face": None,
    "latest_objs": None,
    "latest_pose": None,
    "last_y": None,
    "mouth_dist": 0.0
}

smoothed_hands = {}

# --- MAPPINGS ---
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


def handle_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and 0 <= y <= 40:
        idx = x // 80
        keys = list(config.keys())
        if idx < len(keys):
            k = keys[idx]
            if k == "obj_thresh":
                config[k] = round(config[k] + 0.1, 1) if config[k] < 0.9 else 0.1
            else:
                config[k] = not config[k]


def h_cb(res, img, ts): state["latest_hand"] = res


def f_cb(res, img, ts): state["latest_face"] = res


def o_cb(res, img, ts): state["latest_objs"] = res


def p_cb(res, img, ts): state["latest_pose"] = res


def set_volume_win(change):
    if volume_ctrl:
        try:
            v = np.clip(volume_ctrl.GetMasterVolumeLevelScalar() + (change * 1.5), 0.0, 1.0)
            volume_ctrl.SetMasterVolumeLevelScalar(v, None)
        except:
            pass


cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
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
                running_mode=vision.RunningMode.LIVE_STREAM, score_threshold=0.1, result_callback=o_cb)) as obj_tk, \
            vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='pose_landmarker.task'),
                                             running_mode=vision.RunningMode.LIVE_STREAM,
                                             result_callback=p_cb)) as pose_tk, \
            pyvirtualcam.Camera(width=TARGET_W, height=TARGET_H, fps=30, device=vcam_device) as vcam:

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(cv2.resize(frame, (TARGET_W, TARGET_H)), 1)
            h, w = frame.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int(time.time() * 1000)
            for tk in [hand_tk, face_tk, obj_tk, pose_tk]: tk.detect_async(mp_img, ts)

            curr_time = time.time()
            fps_val = 1 / (curr_time - state["prev_time"])
            state["prev_time"] = curr_time

            out = frame.copy() if config["show_camera"] else np.zeros((h, w, 3), dtype=np.uint8)

            # 1. FACE
            if state["latest_face"] and state["latest_face"].face_landmarks:
                for face_lms in state["latest_face"].face_landmarks:
                    f_pts = np.array([[int(l.x * w), int(l.y * h)] for l in face_lms])
                    if config["show_face"]:
                        for loop in [FACE_OVAL, L_EYE, R_EYE, L_BROW, R_BROW, MOUTH_OUTER, MOUTH_INNER]:
                            cv2.polylines(out, [np.array([f_pts[i] for i in loop], np.int32)], True, (0, 255, 0), 1)
                        if config["show_joints"]:
                            for pt in f_pts: cv2.circle(out, tuple(pt), 1, (0, 255, 0), -1)

                lms = state["latest_face"].face_landmarks[0]
                state["mouth_dist"] = np.linalg.norm(
                    np.array([lms[13].x, lms[13].y]) - np.array([lms[14].x, lms[14].y]))
                is_open = state["mouth_dist"] > 0.045
                if config["mouth_click"]:
                    bar_h = int(np.clip(state["mouth_dist"] * 2000, 0, 150))
                    col = (0, 255, 0) if not is_open else (0, 0, 255)
                    cv2.rectangle(out, (w - 50, 200), (w - 30, 200 - bar_h), col, -1)
                    if is_open and not state["mouth_was_open"]: pyautogui.click()
                state["mouth_was_open"] = is_open

            # 2. HANDS
            if state["latest_hand"] and state["latest_hand"].hand_landmarks:
                for idx, hand_lms in enumerate(state["latest_hand"].hand_landmarks):
                    h_raw = np.array([[l.x, l.y] for l in hand_lms])
                    h_pts = (h_raw * [w, h]).astype(int)
                    if config["show_hand"]:
                        for s, e in HAND_CONN: cv2.line(out, tuple(h_pts[s]), tuple(h_pts[e]), (255, 255, 255), 1)
                        if config["show_joints"]:
                            for pt in h_pts: cv2.circle(out, tuple(pt), 3, (0, 255, 255), -1)

                    if config["vol_gest"]:
                        d = np.linalg.norm(h_raw[4] - h_raw[8])
                        if d < 0.05:
                            if state["last_y"]: set_volume_win(state["last_y"] - h_raw[8][1])
                            state["last_y"] = h_raw[8][1]
                            cv2.circle(out, tuple(h_pts[8]), 10, (0, 0, 255), -1)
                        else:
                            state["last_y"] = None

                    if config["mouse_active"] and idx == 0:
                        off = (h_raw[8] - state["hand_zero_pt"]) * 1.5
                        pyautogui.moveTo(SCREEN_W / 2 + (off[0] * SCREEN_W), SCREEN_H / 2 + (off[1] * SCREEN_H))

            # 3. POSE (BODY)
            if state["latest_pose"] and state["latest_pose"].pose_landmarks:
                for p_lms in state["latest_pose"].pose_landmarks:
                    p_pts = np.array([[int(l.x * w), int(l.y * h)] for l in p_lms])
                    if config["show_body"]:
                        for s, e in POSE_CONN: cv2.line(out, tuple(p_pts[s]), tuple(p_pts[e]), (0, 0, 255), 2)
                        if config["show_joints"]:
                            for pt in p_pts: cv2.circle(out, tuple(pt), 4, (255, 255, 255), -1)

            # 4. OBJECTS
            if config["show_objs"] and state["latest_objs"]:
                for d in state["latest_objs"].detections:
                    if d.categories[0].score >= config["obj_thresh"]:
                        b = d.bounding_box
                        cv2.rectangle(out, (int(b.origin_x), int(b.origin_y)),
                                      (int(b.origin_x + b.width), int(b.origin_y + b.height)), (255, 0, 0), 2)
                        cv2.putText(out, f"{d.categories[0].category_name} {int(d.categories[0].score * 100)}%",
                                    (int(b.origin_x), int(b.origin_y) - 10), 0, 0.5, (255, 0, 0), 2)

            if config["v_cam"]:
                vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                vcam.sleep_until_next_frame()

            display = out.copy()
            if config["debug_mode"]:
                for i, txt in enumerate([f"FPS: {int(fps_val)}", f"CPU: {psutil.cpu_percent()}%",
                                         f"RAM: {psutil.virtual_memory().percent}%"]):
                    cv2.putText(display, txt, (10, h - 20 - (i * 25)), 0, 0.6, (0, 255, 255), 2)

            labels = ["CAM", "FACE", "HAND", "BODY", "OBJ", "JOINT", "MSE", "VCAM", "DEBUG", "M-CLK", "VOL",
                      f"TH:{config['obj_thresh']}"]
            keys = list(config.keys())
            for i, label in enumerate(labels):
                col = (0, 150, 0) if (i < len(keys) and config[keys[i]] == True) else (0, 0, 150)
                if i >= len(keys): col = (50, 50, 50)
                cv2.rectangle(display, (i * 80, 0), (i * 80 + 75, 40), col, -1)
                cv2.putText(display, label, (i * 80 + 5, 25), 0, 0.35, (255, 255, 255), 1)

            cv2.imshow(WINDOW_MAIN, display)
            if cv2.waitKey(1) & 0xFF == 27: break
except Exception as e:
    print(f"Runtime Error: {e}")
finally:
    cap.release(); cv2.destroyAllWindows()