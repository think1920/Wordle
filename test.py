import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
import cv2
import numpy as np
import mediapipe as mp
import base64
import asyncio
import flet as ft
import time
import random
import socket
from typing import Optional

BLANK_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/axn3iQAAAAASUVORK5CYII="

async def safe_update_ctrl(ctrl):
    if hasattr(ctrl, "update_async"):
        await ctrl.update_async()
    else:
        ctrl.update()

LEFT_ROWS  = ["QWZ", "EAS", "DIO", "ULC", "X"]
RIGHT_ROWS = ["PJK", "RTY", "FGH", "VBN", "M"]

def build_vertical_split_keyboards(w, h):
    side_w = int(w * 0.22)
    side_h = int(h * 0.86)
    margin_x = int(w * 0.055)
    margin_y = int(h * 0.06)
    left_x  = margin_x
    right_x = w - margin_x - side_w
    y0 = margin_y
    left_rect  = (left_x,  y0, side_w, side_h)
    right_rect = (right_x, y0, side_w, side_h)

    def build_side(rows, x0, y0, sw, sh):
        row_h = sh // 5
        pad = max(12, sw // 26)
        keys = []
        for r, row in enumerate(rows):
            cols = len(row)
            cell_w = (sw - (cols + 1) * pad) // cols
            cell_h = row_h - 2 * pad
            key_size = max(1, min(cell_w, cell_h))
            y = y0 + r * row_h + pad + (cell_h - key_size) // 2
            for i, ch in enumerate(row):
                cell_x = x0 + pad + i * (cell_w + pad)
                x = cell_x + (cell_w - key_size) // 2
                keys.append({"label": ch, "x": x, "y": y, "w": key_size, "h": key_size})
        return keys

    keys_left  = build_side(LEFT_ROWS,  left_x,  y0, side_w, side_h)
    keys_right = build_side(RIGHT_ROWS, right_x, y0, side_w, side_h)
    return keys_left + keys_right, left_rect, right_rect

def point_in_rect(px, py, rect, margin=0):
    x, y, w, h = rect
    return (x + margin) <= px <= (x + w - margin) and (y + margin) <= py <= (y + h - margin)

def draw_keyboards_vertical(frame_bgr, keys, left_rect, right_rect,
                            hover_label=None, pinch_label=None,
                            highlight_left=False, highlight_right=False):
    for rect, hl in [(left_rect, highlight_left), (right_rect, highlight_right)]:
        x, y, w, h = rect
        overlay = frame_bgr.copy()
        color = (0, 0, 255) if hl else (0, 0, 0)
        alpha = 0.22 if hl else 0.12
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (200, 200, 200), 2)
    for k in keys:
        px, py, pw, ph = k["x"], k["y"], k["w"], k["h"]
        ch = k["label"]
        if pinch_label == ch:
            fill = (40, 160, 255)
        elif hover_label == ch:
            fill = (80, 220, 80)
        else:
            fill = (50, 50, 50)
        cv2.rectangle(frame_bgr, (px, py), (px + pw, py + ph), fill, -1)
        cv2.rectangle(frame_bgr, (px, py), (px + pw, py + ph), (220, 220, 220), 2)
        fs = max(0.7, min(pw, ph) / 40.0)
        (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        tx = px + (pw - tw) // 2
        ty = py + (ph + th) // 2 - 4
        cv2.putText(frame_bgr, ch, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

DRAG_SIZE = 120
PLACED_SIZE = 96

def draw_letter_box(frame_bgr, letter, center_xy, box_size=DRAG_SIZE, color=(0, 200, 0), alpha=0.35):
    x, y = int(center_xy[0]), int(center_xy[1])
    w = h = int(box_size)
    x0, y0 = x - w // 2, y - h // 2
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame_bgr, 0.65, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 2)
    fs = 3.0 if box_size >= 110 else 2.4
    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, fs, 6)
    cv2.putText(frame_bgr, letter, (x - tw // 2, y + th // 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 6, cv2.LINE_AA)

def draw_placed_letters(frame_bgr, letters):
    if not letters:
        return
    overlay = frame_bgr.copy()
    for item in letters:
        x, y = map(int, item["pos"])
        s = int(item.get("size", PLACED_SIZE))
        x0, y0 = x - s // 2, y - s // 2
        cv2.rectangle(overlay, (x0, y0), (x0 + s, y0 + s), (90, 180, 90), -1)
    cv2.addWeighted(overlay, 0.10, frame_bgr, 0.90, 0, frame_bgr)
    for item in letters:
        x, y = map(int, item["pos"])
        s = int(item.get("size", PLACED_SIZE))
        x0, y0 = x - s // 2, y - s // 2
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + s, y0 + s), (220, 255, 220), 2)
        fs = 2.2
        (tw, th), _ = cv2.getTextSize(item["ch"], cv2.FONT_HERSHEY_SIMPLEX, fs, 4)
        cv2.putText(frame_bgr, item["ch"], (x - tw // 2, y + th // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 4, cv2.LINE_AA)

def hit_test_letter(letters, px, py):
    for i in range(len(letters) - 1, -1, -1):
        item = letters[i]
        x, y = item["pos"]
        s = item.get("size", PLACED_SIZE)
        half = s / 2
        if (x - half) <= px <= (x + half) and (y - half) <= py <= (y + half):
            return i
    return -1

def compose_lines(letters, row_tol=None):
    if not letters:
        return [], []
    pts = [{"ch": it["ch"], "x": float(it["pos"][0]), "y": float(it["pos"][1])} for it in letters]
    if row_tol is None:
        row_tol = PLACED_SIZE * 0.75
    pts.sort(key=lambda p: (p["y"], p["x"]))
    rows = []
    for p in pts:
        placed = False
        for row in rows:
            if abs(p["y"] - row["y"]) <= row_tol:
                row["items"].append(p)
                row["y"] = (row["y"] * (len(row["items"])-1) + p["y"]) / len(row["items"])
                placed = True
                break
        if not placed:
            rows.append({"y": p["y"], "items": [p]})
    rows.sort(key=lambda r: r["y"])
    for row in rows:
        row["items"].sort(key=lambda p: p["x"])
    lines = ["".join(p["ch"] for p in row["items"]) for row in rows]
    baselines = [{"y": row["y"], "x_min": min(p["x"] for p in row["items"]),
                  "x_max": max(p["x"] for p in row["items"])} for row in rows]
    return lines, baselines

def pinch_distance(hand):
    lm = hand.landmark
    a, b = lm[4], lm[8]
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
def is_pinching(hand, thr=0.1): return pinch_distance(hand) < thr
def index_pixel(hand, w, h):
    idx = hand.landmark[8]
    return np.array([idx.x * w, idx.y * h], np.float32)

def load_topics(path="topics.txt"):
    topics = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    topics.append(s)
    except FileNotFoundError:
        topics = ["1 môn học", "1 nhạc cụ", "1 con vật", "1 bộ phận cơ thể", "1 ca sĩ", "1 quốc gia", "1 thành phố", "1 món ăn", "1 loài hoa", "1 đồ dùng học tập"]
    return topics

class GameState:
    def __init__(self, topics):
        self.pool = topics[:]
        self.current_topic: Optional[str] = None
        self.round_end_ts = 0.0
        self.in_round = False
        self.last_result = ""
        self.round_secs = 90.0
        self.pre_secs = 3.0
        self.in_ready = False
        self.pre_end_ts = 0.0

    def next_topic_from_host(self) -> Optional[str]:
        if not self.pool:
            return None
        idx = random.randrange(len(self.pool))
        return self.pool.pop(idx)

    def start_ready(self, topic: str, pre_secs=None, round_secs=None):
        self.current_topic = topic
        if pre_secs is not None:
            self.pre_secs = float(pre_secs)
        if round_secs is not None:
            self.round_secs = float(round_secs)
        self.in_ready = True
        self.in_round = False
        self.pre_end_ts = time.time() + self.pre_secs
        self.round_end_ts = 0.0
        self.last_result = ""

    def start_round(self):
        self.in_ready = False
        self.in_round = True
        self.round_end_ts = time.time() + self.round_secs

    def finish_round(self, final_string):
        self.in_round = False
        self.last_result = final_string

    def topics_remaining(self):
        return len(self.pool)

class LetterDragState:
    def __init__(self):
        self.active_letter = None
        self.pos = np.array([0, 0], np.float32)
        self.target_pos = self.pos.copy()
        self.initial_hand_px = None
        self.initial_letter_pos = None
        self.placed_letters = []
        self.hover_label = None
        self.pinch_label = None
        self.was_pinching = False
        self.smooth_alpha = 0.25
        self.kx = self.ky = 1.2
        self.in_left_zone = False
        self.in_right_zone = False

    def clamp(self, p, w, h, m=60):
        return np.array([np.clip(p[0], m, w - m), np.clip(p[1], m, h - m)], np.float32)

    def begin_pick_from_key(self, key):
        self.active_letter = key["label"]
        cx = key["x"] + key["w"] / 2
        cy = key["y"] + key["h"] / 2
        self.pos = np.array([cx, cy], np.float32)
        self.target_pos = self.pos.copy()
        self.initial_letter_pos = self.pos.copy()
        self.pinch_label = key["label"]

    def begin_pick_from_placed(self, item):
        self.active_letter = item["ch"]
        self.pos = item["pos"].copy()
        self.target_pos = self.pos.copy()
        self.initial_letter_pos = self.pos.copy()
        self.pinch_label = None

    def end_drop(self, w, h, left_rect, right_rect):
        if self.active_letter:
            inside_left  = point_in_rect(self.pos[0], self.pos[1], left_rect, 8)
            inside_right = point_in_rect(self.pos[0], self.pos[1], right_rect, 8)
            if not (inside_left or inside_right):
                self.placed_letters.append({
                    "ch": self.active_letter,
                    "pos": self.clamp(self.pos, w, h),
                    "size": PLACED_SIZE
                })
        self.active_letter = None
        self.initial_hand_px = None
        self.initial_letter_pos = None
        self.pinch_label = None
        self.in_left_zone = False
        self.in_right_zone = False

    def update(self, res, keys, left_rect, right_rect, w, h):
        self.hover_label = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            px, py = index_pixel(hand, w, h)
            key = next((k for k in keys if k["x"] <= px <= k["x"] + k["w"] and k["y"] <= py <= k["y"] + k["h"]), None)
            if key:
                self.hover_label = key["label"]
            pinching = is_pinching(hand)
            if pinching and not self.was_pinching and not self.active_letter:
                idx = hit_test_letter(self.placed_letters, px, py)
                if idx >= 0:
                    item = self.placed_letters.pop(idx)
                    self.begin_pick_from_placed(item)
                    self.initial_hand_px = np.array([px, py], np.float32)
                elif key is not None:
                    self.begin_pick_from_key(key)
                    self.initial_hand_px = np.array([px, py], np.float32)
            if pinching and self.active_letter and self.initial_hand_px is not None:
                delta = np.array([px, py], np.float32) - self.initial_hand_px
                self.target_pos = self.clamp(self.initial_letter_pos + delta * np.array([self.kx, self.ky]), w, h)
            if not pinching and self.was_pinching:
                self.end_drop(w, h, left_rect, right_rect)
            self.was_pinching = pinching
        else:
            if self.was_pinching:
                self.end_drop(w, h, left_rect, right_rect)
            self.was_pinching = False
        self.in_left_zone  = point_in_rect(self.pos[0], self.pos[1], left_rect, 8)
        self.in_right_zone = point_in_rect(self.pos[0], self.pos[1], right_rect, 8)
        self.pos = (1 - self.smooth_alpha) * self.pos + self.smooth_alpha * self.target_pos

class NetCfg:
    def __init__(self):
        self.role = "host"                
        self.peer_ip = "127.0.0.1"       
        self.port = 3636             

class ConnState:
    def __init__(self):
        self.ctrl_reader = None
        self.ctrl_writer = None
        self.vid_reader  = None
        self.vid_writer  = None
        self.connected = False

async def open_as_host(port: int) -> ConnState:
    state = ConnState()
    ctrl_ready = asyncio.Event()
    vid_ready = asyncio.Event()

    async def handle_ctrl(reader, writer):
        state.ctrl_reader = reader; state.ctrl_writer = writer
        sock = writer.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        ctrl_ready.set()
        hello = (await reader.readline()).decode("utf-8","ignore").strip()
        if hello == "HELLO":
            writer.write(b"WELCOME\n")
            await writer.drain()

    async def handle_video(reader, writer):
        state.vid_reader = reader; state.vid_writer = writer
        sock = writer.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        vid_ready.set()

    ctrl_server = await asyncio.start_server(handle_ctrl, "0.0.0.0", port+1)
    vid_server  = await asyncio.start_server(handle_video, "0.0.0.0", port)

    try:
        await asyncio.wait_for(ctrl_ready.wait(), timeout=15.0)
        await asyncio.wait_for(vid_ready.wait(), timeout=15.0)
    except asyncio.TimeoutError:
        ctrl_server.close(); vid_server.close()
        await ctrl_server.wait_closed(); await vid_server.wait_closed()
        raise RuntimeError("Loi ket noi client")

    state.connected = True
    ctrl_server.close(); vid_server.close()
    return state

async def open_as_client(ip: str, port: int, timeout=10.0, retries=5) -> ConnState:
    state = ConnState()
    last_err = None
    for _ in range(retries):
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(ip, port+1), timeout=timeout)
            state.ctrl_reader, state.ctrl_writer = reader, writer
            writer.write(b"HELLO\n")
            await writer.drain()
            line = (await reader.readline()).decode("utf-8", "ignore").strip()
            if line != "WELCOME":
                raise RuntimeError("Khong nhan duoc msg")

            vr, vw = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=timeout)
            state.vid_reader, state.vid_writer = vr, vw

            for w in (state.ctrl_writer, state.vid_writer):
                sock = w.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            state.connected = True
            return state
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.8) 
    raise last_err


async def video_send_loop(writer: asyncio.StreamWriter, frame_q: asyncio.Queue, stop_evt: asyncio.Event):
    try:
        while not stop_evt.is_set():
            try:
                jpg = await asyncio.wait_for(frame_q.get(), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            if jpg is None:
                continue
            n = len(jpg)
            writer.write(n.to_bytes(4, "big") + jpg)
            await writer.drain()
    except Exception:
        pass

async def video_recv_loop(reader: asyncio.StreamReader, set_peer_frame, stop_evt: asyncio.Event):
    try:
        while not stop_evt.is_set():
            hdr = await reader.readexactly(4)
            n = int.from_bytes(hdr, "big")
            if n <= 0 or n > 20_000_000:
                continue
            buf = await reader.readexactly(n)
            set_peer_frame(buf)
    except Exception:
        pass

async def ctrl_recv_loop(reader: asyncio.StreamReader, on_msg, stop_evt: asyncio.Event):
    try:
        while not stop_evt.is_set():
            line = await reader.readline()
            if not line:
                break
            s = line.decode("utf-8", "ignore").strip()
            await on_msg(s)
    except Exception:
        pass

def encode_jpeg_for_stream(bgr):
    try:
        h, w = bgr.shape[:2]
        target_w = 960 
        if w > target_w:
            scale = target_w / w
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return None

async def video_loop(page: ft.Page,
                     my_img: ft.Image, peer_img: ft.Image,
                     debug: ft.Text, topic_text: ft.Text, timer_text: ft.Text,
                     my_ans_text: ft.Text, peer_ans_text: ft.Text, verdict_text: ft.Text,
                     next_btn: ft.ElevatedButton, status_text: ft.Text,
                     role_dd: ft.Dropdown, ip_field: ft.TextField, port_field: ft.TextField, connect_btn: ft.ElevatedButton):
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        debug.value = "Không mở được camera."
        await safe_update_ctrl(debug)
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(False, 1, 0, 0.6, 0.5)

    topics = load_topics("topics.txt")
    game = GameState(topics)
    state = LetterDragState()

    my_result_str = ""
    peer_result_str = ""

    def _normalize(s: str) -> str:
        return "".join(s.split()).upper()

    async def _update_results_ui():
        my_ans_text.value = f"Your answer: {my_result_str}"
        peer_ans_text.value = f"Peer answer: {peer_result_str}"
        if my_result_str and peer_result_str:
            if _normalize(my_result_str) == _normalize(peer_result_str):
                verdict_text.value = "Congratulation !"
                verdict_text.color = ft.colors.GREEN
            else:
                verdict_text.value = "Not pass"
                verdict_text.color = ft.colors.RED
        else:
            verdict_text.value = ""
            verdict_text.color = None
        await safe_update_ctrl(my_ans_text)
        await safe_update_ctrl(peer_ans_text)
        await safe_update_ctrl(verdict_text)

    def _clear_screen_letters():
        state.placed_letters.clear()
        state.active_letter = None
        state.pinch_label = None

    net = NetCfg()
    conn = ConnState()
    frame_q = asyncio.Queue(maxsize=2)
    stop_evt = asyncio.Event()
    peer_frame = {"buf": None, "ts": 0.0}

    def set_peer_frame(buf: bytes):
        peer_frame["buf"] = buf
        peer_frame["ts"] = time.time()

    async def on_ctrl_msg(s: str):
        nonlocal my_result_str, peer_result_str

        if s == "HELLO" and net.role == "host":
            conn.ctrl_writer.write(b"WELCOME\n")
            await conn.ctrl_writer.drain()
            return
        if s == "WELCOME" and net.role == "client":
            return

        if s.startswith("TOPIC|"):
            toks = s.split("|")
            t = "|".join(toks[1:-2])
            pre, rnd = toks[-2], toks[-1]
            game.start_ready(t, pre_secs=float(pre), round_secs=float(rnd))

            state.placed_letters.clear()
            state.active_letter = None
            state.pinch_label = None
            my_result_str = ""
            peer_result_str = ""

            topic_text.value = f"Chủ đề: {t}"
            timer_text.value = f"Chuẩn bị: {int(game.pre_secs)}s"
            page.update()
            await _update_results_ui()
            return

        if s == "START":
            game.start_round()
            return

        if s.startswith("RESULT|"):
            peer_result_str = s[len("RESULT|"):]
            await _update_results_ui()
            return

    async def do_connect():
        nonlocal conn
        if conn.connected:
            return
        try:
            net.role = role_dd.value
            net.peer_ip = ip_field.value.strip()
            net.port = int(port_field.value.strip())

            if net.role == "host":
                status_text.value = f"Listening on {net.port} (video) và {net.port+1} (control). Chờ client..."
                await safe_update_ctrl(status_text)
                conn = await open_as_host(net.port)
            else:
                status_text.value = "Đang kết nối host..."
                await safe_update_ctrl(status_text)
                conn = await open_as_client(net.peer_ip, net.port, timeout=10.0, retries=5)

            connect_btn.disabled = True
            role_dd.disabled = True
            ip_field.disabled = (net.role == "client")
            port_field.disabled = True
            await safe_update_ctrl(connect_btn); await safe_update_ctrl(role_dd)
            await safe_update_ctrl(ip_field); await safe_update_ctrl(port_field)

            page.run_task(video_send_loop, conn.vid_writer, frame_q, stop_evt)
            page.run_task(video_recv_loop, conn.vid_reader, set_peer_frame, stop_evt)
            page.run_task(ctrl_recv_loop, conn.ctrl_reader, on_ctrl_msg, stop_evt)

            status_text.value = "Connected"
            next_btn.disabled = (net.role != "host")
            await safe_update_ctrl(status_text); await safe_update_ctrl(next_btn)
        except Exception as e:
            status_text.value = f"Lỗi kết nối: {e}"
            await safe_update_ctrl(status_text)

    def on_connect(e):
        page.run_task(do_connect)
    connect_btn.on_click = on_connect

    def on_next(e):
        nonlocal my_result_str, peer_result_str

        if not conn.connected or net.role != "host":
            return

        topic = game.next_topic_from_host()
        if not topic:
            topic_text.value = "Hết chủ đề"
            timer_text.value = "Hết chủ đề"
            page.update()
            return

        game.start_ready(topic, pre_secs=3.0, round_secs=90.0)

        state.placed_letters.clear()
        state.active_letter = None
        state.pinch_label = None
        my_result_str = ""
        peer_result_str = ""

        topic_text.value = f"Chủ đề: {topic}"
        timer_text.value = f"Chuẩn bị: {int(game.pre_secs)}s"
        page.update()
        page.run_task(_update_results_ui)

        async def broadcast():
            conn.ctrl_writer.write(f"TOPIC|{topic}|{game.pre_secs}|{game.round_secs}\n".encode("utf-8"))
            await conn.ctrl_writer.drain()
            await asyncio.sleep(0.1)
            conn.ctrl_writer.write(b"START\n")
            await conn.ctrl_writer.drain()
        page.run_task(broadcast)


    next_btn.on_click = on_next
    peer_img.src_base64 = BLANK_PNG_BASE64
    await safe_update_ctrl(my_img)
    await safe_update_ctrl(peer_img)

    t0 = time.time()
    last_ui = 0.0
    last_sent = 0.0

    while page.session is not None:
        ok, frame = cap.read()
        if not ok:
            await asyncio.sleep(0.02)
            continue
        frame = cv2.flip(frame, 1)
        send_bgr = frame.copy()
        h, w = frame.shape[:2]
        keys, left_rect, right_rect = build_vertical_split_keyboards(w, h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if not conn.connected:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            (tw, th), _ = cv2.getTextSize("Waiting for connection...", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(frame, "Waiting for connection...", (w//2 - tw//2, h//2 + th//3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        else:
            if game.in_round:
                state.update(res, keys, left_rect, right_rect, w, h)
            else:
                state.in_left_zone  = point_in_rect(state.pos[0], state.pos[1], left_rect, 8)
                state.in_right_zone = point_in_rect(state.pos[0], state.pos[1], right_rect, 8)

            draw_placed_letters(frame, state.placed_letters)
            if state.active_letter:
                color = (0, 0, 255) if (state.in_left_zone or state.in_right_zone) else (0, 200, 0)
                draw_letter_box(frame, state.active_letter, state.pos, box_size=DRAG_SIZE, color=color)
            draw_keyboards_vertical(frame, keys, left_rect, right_rect,
                                    hover_label=state.hover_label,
                                    pinch_label=state.pinch_label,
                                    highlight_left=state.in_left_zone,
                                    highlight_right=state.in_right_zone)

            lines, _ = compose_lines(state.placed_letters, row_tol=PLACED_SIZE*0.75)
            now = time.time()

            if game.in_ready and not game.in_round:
                remain = max(0, int(np.ceil(game.pre_end_ts - now)))
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                fs = max(2.5, min(w, h) / 240)
                msg = f"{remain}"
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs*3, 10)
                cv2.putText(frame, msg, (w//2 - tw//2, h//2 + th//3),
                            cv2.FONT_HERSHEY_SIMPLEX, fs*3, (255,255,255), 10, cv2.LINE_AA)
                timer_text.value = f"Chuẩn bị: {remain}s"
                await safe_update_ctrl(timer_text)
                if now >= game.pre_end_ts:
                    game.start_round()

            elif game.in_round:
                tleft = max(0.0, game.round_end_ts - now)
                if tleft <= 0.0:
                    final_str = "".join(lines)
                    game.finish_round(final_str)

                    my_result_str = final_str
                    await _update_results_ui()

                    state.placed_letters.clear()
                    state.active_letter = None
                    state.pinch_label = None

                    async def _send_result():
                        msg = f"RESULT|{final_str}\n".encode("utf-8")
                        conn.ctrl_writer.write(msg)
                        await conn.ctrl_writer.drain()
                    page.run_task(_send_result)


                    timer_text.value = "Hết giờ"
                    next_btn.disabled = (net.role != "host")
                    page.update()
                else:
                    timer_text.value = f"Thời gian: {int(np.ceil(tleft))}s"
                    await safe_update_ctrl(timer_text)

            if (not game.in_round) and (game.current_topic is None) and (game.topics_remaining() == 0):
                timer_text.value = "Hết chủ đề"
                await safe_update_ctrl(timer_text)

        now = time.time()
        if now - last_sent >= 1.0 / 24.0 and conn.connected:
            jpg = encode_jpeg_for_stream(send_bgr)
            if jpg:
                if frame_q.full():
                    try:
                        _ = frame_q.get_nowait()
                    except Exception:
                        pass
                try:
                    frame_q.put_nowait(jpg)
                except Exception:
                    pass
            last_sent = now

        ok_my, buf_my = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok_my:
            my_img.src_base64 = base64.b64encode(buf_my).decode("ascii")
            await safe_update_ctrl(my_img)

        if peer_frame["buf"]:
            try:
                arr = np.frombuffer(peer_frame["buf"], dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    ok_p, buf_p = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if ok_p:
                        peer_img.src_base64 = base64.b64encode(buf_p).decode("ascii")
                        await safe_update_ctrl(peer_img)
            except Exception:
                pass

        if now - last_ui >= 0.5:
            debug.value = f"Role {net.role.upper()} | Connected: {conn.connected} | Placed: {len(state.placed_letters)}"
            topic_text.value = f"Chủ đề: {game.current_topic or '-'}"
            await safe_update_ctrl(debug)
            await safe_update_ctrl(topic_text)
            last_ui = now
        await asyncio.sleep(0.005)

    stop_evt.set()
    try:
        if conn.ctrl_writer: conn.ctrl_writer.close()
        if conn.vid_writer:  conn.vid_writer.close()
    except Exception:
        pass
    cap.release()

def main(page: ft.Page):
    page.title = "Word Duel"
    page.window_width = 1320
    page.window_height = 900
    page.theme_mode = "dark"
    page.padding = 10

    my_img = ft.Image(width=960, height=540, src_base64=BLANK_PNG_BASE64, fit=ft.ImageFit.CONTAIN, gapless_playback=True)
    peer_img = ft.Image(width=320, height=180, src_base64=BLANK_PNG_BASE64, fit=ft.ImageFit.CONTAIN, gapless_playback=True)
    t_debug  = ft.Text("Starting...", size=12)
    t_topic  = ft.Text("Chủ đề: ...", size=16, weight=ft.FontWeight.BOLD)
    t_timer  = ft.Text("Thời gian: 90s", size=16)
    t_my_ans   = ft.Text("Your answer: ", size=14)
    t_peer_ans = ft.Text("Peer answer: ", size=14)
    t_verdict  = ft.Text("", size=16, weight=ft.FontWeight.BOLD)
    t_status = ft.Text("Not connected", size=12)

    role_dd   = ft.Dropdown(label="Role", value="host", options=[ft.dropdown.Option("host"), ft.dropdown.Option("client")], width=120)
    ip_field  = ft.TextField(label="Host IP", value="127.0.0.1", width=220, dense=True)
    port_field= ft.TextField(label="Port", value="3636", width=120, dense=True)
    connect_btn = ft.ElevatedButton("Kết nối")

    next_btn = ft.ElevatedButton("Tiếp tục", disabled=True)  

    page.add(
        ft.Column(
            [
                ft.Row([role_dd, ip_field, port_field, connect_btn, t_status], alignment=ft.MainAxisAlignment.START, spacing=12),
                ft.Row([my_img, ft.Column([peer_img, next_btn, t_topic, t_timer, t_my_ans, t_peer_ans, t_verdict, t_debug], spacing=10)], alignment=ft.MainAxisAlignment.START, spacing=16),
                # ft.Text("Game on")
            ],
            spacing=10
        )
    )

    page.run_task(
        video_loop,
        page, my_img, peer_img, t_debug, t_topic, t_timer,
        t_my_ans, t_peer_ans, t_verdict,   
        next_btn, t_status, role_dd, ip_field, port_field, connect_btn
    )

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.FLET_APP)