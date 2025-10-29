import asyncio
import random
from pathlib import Path
from collections import Counter
import subprocess
import sys
import json
from typing import Optional
import flet as ft
import flet_audio as fta
from contextlib import suppress

ROWS = 6
COLS = 5

ROOT = Path(__file__).parent
ALLOWED_FILE = ROOT / "wordle-allowed-guesses.txt"
ANSWERS_FILE = ROOT / "wordle-answers-alphabetical.txt"

def load_wordlists():
    def read_words(p: Path):
        with p.open("r", encoding="utf-8") as f:
            return [w.strip().lower() for w in f if w.strip() and len(w.strip()) == COLS]
    allowed = set(read_words(ALLOWED_FILE))
    answers = read_words(ANSWERS_FILE)
    allowed |= set(answers)
    return allowed, answers

ALLOWED, ANSWERS = load_wordlists()

def score_guess(answer: str, guess: str):
    res = ["b"] * COLS
    cnt = Counter(answer)
    for i, (a, g) in enumerate(zip(answer, guess)):
        if g == a:
            res[i] = "g"
            cnt[g] -= 1
    for i, g in enumerate(guess):
        if res[i] == "b" and cnt[g] > 0:
            res[i] = "y"
            cnt[g] -= 1
    return res

class WordleGame:
    def __init__(self):
        self.answer = random.choice(ANSWERS)
        self.row = 0
        self.col = 0
        self.board = [["" for _ in range(COLS)] for _ in range(ROWS)]
        self.state = [["" for _ in range(COLS)] for _ in range(ROWS)]
        self.kb_state = {}
        self.hint_used = False

    def reset(self):
        self.__init__()

    def current_guess(self):
        return "".join(self.board[self.row])

    def apply_feedback(self, guess, fb):
        self.state[self.row] = fb
        rank = {"": 0, "b": 1, "y": 2, "g": 3}
        for ch, tok in zip(guess, fb):
            if rank.get(tok, 0) > rank.get(self.kb_state.get(ch, ""), 0):
                self.kb_state[ch] = tok

class TcpSession:
    def __init__(self, page: "ft.Page"):
        self.page = page
        self.reader = None
        self.writer = None
        self.read_task = None
        self.on_message = None
        self.connected = False
        self.server = None
        self.server_task = None

    async def host(self, host: str, port: int):
        fut = asyncio.get_running_loop().create_future()

        async def _handle(reader, writer):
            self.reader, self.writer = reader, writer
            self.connected = True
            if not fut.done():
                fut.set_result(True)
            self.read_task = asyncio.create_task(self._read_loop())

        self.server = await asyncio.start_server(_handle, host, port)
        await fut
        self.server_task = asyncio.create_task(self.server.serve_forever())

    async def connect(self, host: str, port: int):
        self.reader, self.writer = await asyncio.open_connection(host, port)
        self.connected = True
        self.read_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        try:
            while True:
                line = await self.reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8").strip())
                except Exception:
                    continue
                if self.on_message:
                    await self.on_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self.connected = False

    def send(self, obj: dict):
        if not self.writer:
            return
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self.writer.write(data)

    async def close(self):
        if self.read_task:
            self.read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.read_task
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
        if self.server:
            try:
                self.server.close()
                await self.server.wait_closed()
            except Exception:
                pass
            self.server = None
        if self.server_task:
            self.server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.server_task
            self.server_task = None
        self.connected = False

def choose_shared_answer_idx():
    return random.randrange(len(ANSWERS))

def main(page: "ft.Page"):
    page.title = "Wordle"
    page.window.width = 1440
    page.window.height = 900
    page.theme_mode = ft.ThemeMode.DARK

    body = ft.Container(expand=True, alignment=ft.alignment.top_center)

    def render_multiplayer(page: "ft.Page", body: ft.Container, show_menu):
        role_dd = ft.Dropdown(
            value="host",
            options=[ft.dropdown.Option("host"), ft.dropdown.Option("client")],
            width=140,
        )

        def focus_board():
            try:
                page.set_focus(my_grid_box)
            except:
                pass

        def ghost_btn(label_text: str, on_click, tooltip: str = "", width: int = 90, height: int = 40):
            lbl_ref = ft.Ref[ft.Text]()
            base = ft.Container(
                content=ft.Text(label_text, ref=lbl_ref, size=12, weight=ft.FontWeight.W_600),
                width=width,
                height=height,
                alignment=ft.alignment.center,
                bgcolor=ft.Colors.WHITE10,
                border=ft.border.all(1, ft.Colors.WHITE24),
                border_radius=6,
                tooltip=tooltip,
                ink=False,
            )
            hit = ft.Container(
                width=width,
                height=height,
                bgcolor=ft.Colors.TRANSPARENT,
                on_click=lambda e: (on_click(e), focus_board()),
            )
            return ft.Stack([base, hit]), lbl_ref

        ip_field = ft.TextField(label="Host IP", value="127.0.0.1", width=190)
        port_field = ft.TextField(label="Port", value="3636", width=110)
        connect_btn, _ = ghost_btn("Kết nối", lambda e: page.run_task(do_connect), width=120)
        back_btn, _    = ghost_btn("← Menu",  lambda e: page.run_task(go_back),  width=120)

        status_text = ft.Text("Chọn vai trò, nhập IP/Port rồi Kết nối", size=12, opacity=0.9)

        my_game = WordleGame()
        peer_game = WordleGame()
        shared_idx = {"v": None}

        started = {"v": False}
        me_win = {"v": False}
        me_fail = {"v": False}
        peer_win = {"v": False}
        peer_fail = {"v": False}
        hint_lock_owner = {"v": None}
        my_can_hint = {"v": True}
        peer_hint_info = ft.Text("", size=12, opacity=0.8)

        my_tiles, my_row_boxes, my_grid_rows = [], [], []
        peer_tiles, peer_row_boxes, peer_grid_rows = [], [], []

        def tile_bg(tok: str):
            if tok == "g":
                return ft.Colors.GREEN
            if tok == "y":
                return ft.Colors.AMBER
            if tok == "b":
                return ft.Colors.BLUE_GREY
            return ft.Colors.WHITE10

        def make_cell(text_val: str = "", tok: str = ""):
            return ft.Container(
                content=ft.Text(text_val, weight=ft.FontWeight.BOLD, size=24),
                width=56,
                height=56,
                alignment=ft.alignment.center,
                bgcolor=tile_bg(tok),
                border_radius=8,
                border=ft.border.all(1, ft.Colors.WHITE24),
            )

        def rebuild_grids():
            my_tiles.clear(); my_row_boxes.clear(); my_grid_rows.clear()
            peer_tiles.clear(); peer_row_boxes.clear(); peer_grid_rows.clear()

            for _ in range(ROWS):
                row_tiles = [make_cell("", "") for _ in range(COLS)]
                for cont in row_tiles:
                    my_tiles.append((cont.content, cont))
                row_box = ft.Container(
                    content=ft.Row(row_tiles, alignment=ft.MainAxisAlignment.CENTER, spacing=6),
                    offset=ft.Offset(0, 0),
                    animate_offset=ft.Animation(120, "easeInOut"),
                )
                my_row_boxes.append(row_box)
                my_grid_rows.append(row_box)

            for _ in range(ROWS):
                row_tiles = [make_cell("", "") for _ in range(COLS)]
                for cont in row_tiles:
                    peer_tiles.append((cont.content, cont))
                row_box = ft.Container(
                    content=ft.Row(row_tiles, alignment=ft.MainAxisAlignment.CENTER, spacing=6)
                )
                peer_row_boxes.append(row_box)
                peer_grid_rows.append(row_box)

        rebuild_grids()

        my_status_text = ft.Text("Gõ chữ. Enter để submit. Backspace để xóa.", size=12)
        my_hint_text = ft.Text("", size=12, opacity=0.9)
        peer_typing_text = ft.Text("Peer typing: ", size=12, opacity=0.9)

        kb_layout = [list("qwertyuiop"), list("asdfghjkl"), ["enter"] + list("zxcvbnm") + ["del"]]
        kb_tiles, kb_rows_ui = [], []

        def kb_color(ch: str):
            tok = my_game.kb_state.get(ch, "")
            return tile_bg(tok)

        def key_tile(label_text: str, key_value: str):
            bg = kb_color(key_value) if len(key_value) == 1 else ft.Colors.GREY_800
            w = 42 if len(key_value) == 1 else 78
            t = ft.Text(label_text, size=12, weight=ft.FontWeight.W_600)
            tile = ft.Container(
                data=key_value,
                content=t,
                width=w,
                height=40,
                alignment=ft.alignment.center,
                bgcolor=bg,
                border_radius=6,
                border=ft.border.all(1, ft.Colors.WHITE24),
                on_click=lambda e: press_key(e.control.data),
            )
            return tile

        for row in kb_layout:
            row_controls = []
            for key in row:
                label = key.upper() if len(key) == 1 else ("ENTER" if key == "enter" else "DEL")
                tile = key_tile(label, key)
                kb_tiles.append(tile)
                row_controls.append(tile)
            kb_rows_ui.append(ft.Row(row_controls, alignment=ft.MainAxisAlignment.CENTER, spacing=6))

        my_grid_box = ft.Container(content=ft.Column(my_grid_rows, spacing=6, alignment=ft.MainAxisAlignment.CENTER))
        peer_grid_box = ft.Container(content=ft.Column(peer_grid_rows, spacing=6, alignment=ft.MainAxisAlignment.CENTER))

        async def _connect_and_refocus():
            await do_connect()
            await refocus_later()

        ip_field.on_submit   = lambda e: page.run_task(_connect_and_refocus)
        port_field.on_submit = lambda e: page.run_task(_connect_and_refocus)

        def refresh_my_board():
            idx = 0
            for r in range(ROWS):
                for c in range(COLS):
                    t_ctrl, cont = my_tiles[idx]
                    t_ctrl.value = my_game.board[r][c].upper()
                    cont.bgcolor = tile_bg(my_game.state[r][c])
                    idx += 1

        def refresh_peer_board():
            idx = 0
            for r in range(ROWS):
                for c in range(COLS):
                    t_ctrl, cont = peer_tiles[idx]
                    t_ctrl.value = peer_game.board[r][c].upper()
                    cont.bgcolor = tile_bg(peer_game.state[r][c])
                    idx += 1

        def color_keyboard():
            for tile in kb_tiles:
                key_val = tile.data
                if key_val and len(key_val) == 1:
                    tile.bgcolor = kb_color(key_val)

        async def shake_current_row():
            idx = min(max(my_game.row, 0), ROWS - 1)
            if idx >= len(my_row_boxes):
                return
            rb = my_row_boxes[idx]
            rb.offset = ft.Offset(0.06, 0); page.update()
            await asyncio.sleep(0.10)
            rb.offset = ft.Offset(-0.06, 0); page.update()
            await asyncio.sleep(0.08)
            rb.offset = ft.Offset(0, 0)
            if hasattr(page, "update_async"):
                await page.update_async()
            else:
                page.update()

        def green_positions(game: WordleGame):
            greens = set()
            for r in range(ROWS):
                for c in range(COLS):
                    if game.state[r][c] == "g":
                        greens.add(c)
            return greens

        def make_hint_pattern():
            import random as _rnd
            greens = green_positions(my_game)
            candidates = [i for i in range(COLS) if i not in greens]
            idx = _rnd.choice(candidates) if candidates else _rnd.randrange(COLS)
            pattern = "".join(my_game.answer[i].upper() if i == idx else "⬛" for i in range(COLS))
            return f"Hint: {pattern}"

        session = TcpSession(page)

        async def on_net_message(msg: dict):
            t = msg.get("t")
            if t == "hello":
                return
            if t == "seed":
                idx = int(msg["idx"])
                shared_idx["v"] = idx
                my_game.answer = ANSWERS[idx]
                peer_game.answer = ANSWERS[idx]
                started["v"] = True
                status_text.value = "Đã đồng bộ đáp án. Bắt đầu chơi."
                page.update(); return
            if t == "typing":
                row = int(msg["row"]); guess = msg["guess"]
                if 0 <= row < ROWS:
                    peer_game.board[row] = list(guess.ljust(COLS))[:COLS]
                    refresh_peer_board()
                    peer_typing_text.value = f"Peer typing: {guess.upper()}"
                    page.update(); return
            if t == "submit":
                row = int(msg["row"]); guess = msg["guess"]
                if 0 <= row < ROWS:
                    fb = score_guess(my_game.answer, guess)
                    peer_game.state[row] = fb
                    peer_game.board[row] = list(guess)[:COLS]
                    refresh_peer_board(); page.update(); return
            if t == "hint":
                if hint_lock_owner["v"] is None:
                    hint_lock_owner["v"] = "peer"
                    my_can_hint["v"] = False
                peer_hint_info.value = "Peer dùng hint"; page.update(); return
            if t == "win":
                peer_win["v"] = True
                status_text.value = "Đối phương đã đoán đúng."; page.update(); return
            if t == "fail":
                peer_fail["v"] = True
                if me_fail["v"] and peer_fail["v"]:
                    _show_answer_row()
                page.update(); return

        session.on_message = on_net_message

        def press_key(ch: str):
            if not started["v"] or me_win["v"] or me_fail["v"]:
                return
            if ch == "enter":
                on_enter(); return
            if ch == "del":
                on_backspace(); return
            if my_game.row >= ROWS:
                return
            if my_game.col < COLS:
                my_game.board[my_game.row][my_game.col] = ch
                my_game.col += 1
                refresh_my_board(); page.update()
                session.send({"t": "typing", "row": my_game.row, "guess": my_game.current_guess()})

        def on_backspace():
            if not started["v"] or me_win["v"] or me_fail["v"]:
                return
            if my_game.row >= ROWS:
                return
            if my_game.col > 0:
                my_game.col -= 1
                my_game.board[my_game.row][my_game.col] = ""
                refresh_my_board(); page.update()
                session.send({"t": "typing", "row": my_game.row, "guess": my_game.current_guess()})

        def _show_answer_row():
            def make_ans_row():
                ans_tiles = []
                for i in range(COLS):
                    ch = my_game.answer[i].upper()
                    ans_tiles.append(
                        ft.Container(
                            content=ft.Text(ch, weight=ft.FontWeight.BOLD, size=24),
                            width=56, height=56,
                            alignment=ft.alignment.center,
                            bgcolor=ft.Colors.GREEN,
                            border_radius=8,
                            border=ft.border.all(1, ft.Colors.WHITE24),
                        )
                    )
                return ft.Container(
                    content=ft.Row(ans_tiles, alignment=ft.MainAxisAlignment.CENTER, spacing=6)
                )

            my_answer_row = make_ans_row()
            peer_answer_row = make_ans_row()

            my_grid_rows.append(my_answer_row)
            my_grid_box.content = ft.Column(my_grid_rows, spacing=6, alignment=ft.MainAxisAlignment.CENTER)

            peer_grid_rows.append(peer_answer_row)
            peer_grid_box.content = ft.Column(peer_grid_rows, spacing=6, alignment=ft.MainAxisAlignment.CENTER)

            page.update()

        def invalidate(msg: str):
            my_status_text.value = msg; page.update()

        def on_enter():
            if not started["v"] or me_win["v"] or me_fail["v"]:
                return
            if my_game.row < ROWS and my_game.col < COLS:
                invalidate("5 chữ cái")
                page.run_task(shake_current_row); return
            if my_game.row >= ROWS:
                invalidate("Bạn đã hết lượt"); return

            guess = my_game.current_guess()
            if guess not in ALLOWED:
                invalidate("Từ không hợp lệ")
                page.run_task(shake_current_row); return

            fb = score_guess(my_game.answer, guess)
            my_game.apply_feedback(guess, fb)
            my_game.col = 0
            refresh_my_board(); color_keyboard(); page.update()
            session.send({"t": "submit", "row": my_game.row, "guess": guess})

            if guess == my_game.answer:
                me_win["v"] = True
                my_status_text.value = "Đúng. Bạn thắng."
                session.send({"t": "win"}); page.update(); return

            my_game.row += 1
            if my_game.row == ROWS:
                me_fail["v"] = True
                my_status_text.value = "Bạn đã hết lượt."
                page.update()
                session.send({"t": "fail"})
                if peer_fail["v"]:
                    _show_answer_row()

        def on_hint_click(_):
            if not started["v"]:
                return
            if hint_lock_owner["v"] is None:
                hint_lock_owner["v"] = "me"
                my_can_hint["v"] = False
                session.send({"t": "hint"})
                my_hint_text.value = make_hint_pattern()
                page.update()
            else:
                invalidate("Hint đã bị khóa bởi bên còn lại")

        async def do_connect(_=None):
            try:
                host = ip_field.value.strip()
                port = int(port_field.value.strip())
                if role_dd.value == "host":
                    status_text.value = "Đang chờ client..."; page.update()
                    await session.host("0.0.0.0", port)
                    idx = choose_shared_answer_idx()
                    shared_idx["v"] = idx
                    my_game.answer = ANSWERS[idx]
                    peer_game.answer = ANSWERS[idx]
                    session.send({"t": "seed", "idx": idx})
                    status_text.value = "Đã kết nối. Bắt đầu chơi."
                    started["v"] = True
                    # ==== DISABLE INPUTS SAU KHI KẾT NỐI ====
                    role_dd.disabled = True
                    ip_field.disabled = True
                    port_field.disabled = True
                    if hasattr(connect_btn, "controls"):  # nếu là ghost_btn
                        for c in connect_btn.controls:
                            c.disabled = True
                    else:
                        connect_btn.disabled = True
                    page.update()
                else:
                    status_text.value = "Đang kết nối tới host..."; page.update()
                    await session.connect(host, port)
                    session.send({"t": "hello"})
                    status_text.value = "Đã kết nối. Chờ đồng bộ đáp án..."
                    # ==== DISABLE INPUTS SAU KHI KẾT NỐI ====
                    role_dd.disabled = True
                    ip_field.disabled = True
                    port_field.disabled = True
                    if hasattr(connect_btn, "controls"):
                        for c in connect_btn.controls:
                            c.disabled = True
                    else:
                        connect_btn.disabled = True
                    page.update()
            except Exception as ex:
                status_text.value = f"Lỗi kết nối: {ex}"; page.update()


        async def go_back(_=None):
            try:
                await session.close()
            except:
                pass

            role_dd.disabled = False
            ip_field.disabled = False
            port_field.disabled = False
            if hasattr(connect_btn, "controls"):
                for c in connect_btn.controls:
                    c.disabled = False
            else:
                connect_btn.disabled = False
            page.update()

            page.on_keyboard_event = None
            show_menu()

        hint_btn, _  = ghost_btn("Hint", on_hint_click, width=90)

        def on_key(e: ft.KeyboardEvent):
            k = e.key
            if k in ("Enter", "NumpadEnter"):
                on_enter()
            elif k in ("Backspace", "Delete"):
                on_backspace()
            else:
                ch = (k or "").lower()
                if len(ch) == 1 and ch.isalpha():
                    press_key(ch)

        page.on_keyboard_event = on_key

        header = ft.Row(
            [
                back_btn,
                ft.Text("Wordle Chơi đôi", size=20, weight=ft.FontWeight.W_700),
                ft.Container(expand=True),
                ft.Text("Vai trò:", size=12), role_dd,
                ip_field, port_field, connect_btn,
            ],
            alignment=ft.MainAxisAlignment.START,
        )

        left_col = ft.Column(
            [
                ft.Text("Bạn", size=14, weight=ft.FontWeight.W_700),
                my_grid_box, ft.Divider(opacity=0),
                my_status_text, my_hint_text, ft.Divider(opacity=0),
                *kb_rows_ui,
                ft.Row([hint_btn, ft.Text(f"Số lượng từ: {len(ANSWERS)}", size=12, opacity=0.6)]),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )

        right_col = ft.Column(
            [
                ft.Text("Đối phương", size=14, weight=ft.FontWeight.W_700),
                peer_grid_box, ft.Divider(opacity=0),
                peer_typing_text, peer_hint_info, status_text,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )

        two_cols = ft.Row(
            [
                ft.Container(content=left_col, expand=True, padding=6),
                ft.VerticalDivider(width=1, opacity=0.2),
                ft.Container(content=right_col, expand=True, padding=6),
            ],
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

        body.content = ft.Column([header, two_cols], spacing=8)
        page.update()

    def show_menu():
        page.on_keyboard_event = None
        stop_music()

        title = ft.Text("Menu", size=26, weight=ft.FontWeight.W_700)
        subtitle = ft.Text("Chọn chế độ chơi", size=14, opacity=0.8)

        btn_single = ft.ElevatedButton(
            "Chơi đơn",
            width=280,
            height=44,
            on_click=lambda e: render_single_player(),
        )
        
        btn_multi = ft.ElevatedButton(
            "Chơi đôi",
            width=280,
            height=44,
            on_click=lambda e: render_multiplayer(page, body, show_menu),
        )

        btn_special = ft.ElevatedButton(
            "Special Game",
            width=280,
            height=44,
            on_click=run_special_game,
        )

        body.content = ft.Column(
            [
                ft.Container(height=24),
                title,
                subtitle,
                ft.Container(height=16),
                btn_single,
                btn_multi,
                btn_special,
                ft.Container(height=8),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        )
        page.update()

    music_ref = ft.Ref[fta.Audio]()

    def stop_music():
        try:
            if music_ref.current:
                try:
                    music_ref.current.on_state_changed = None
                except:
                    pass
                try:
                    music_ref.current.stop()
                except:
                    try:
                        music_ref.current.pause()
                    except:
                        pass
                try:
                    if music_ref.current in page.overlay:
                        page.overlay.remove(music_ref.current)
                except:
                    pass
                music_ref.current = None
            page.update()
        except:
            pass

    def render_single_player():
        game = WordleGame()

        tiles = []
        row_boxes = []
        grid_rows = []

        def tile_bg(tok: str):
            if tok == "g":
                return ft.Colors.GREEN
            if tok == "y":
                return ft.Colors.AMBER
            if tok == "b":
                return ft.Colors.BLUE_GREY
            return ft.Colors.WHITE10

        def make_cell(text_val: str = "", tok: str = ""):
            return ft.Container(
                content=ft.Text(text_val, weight=ft.FontWeight.BOLD, size=24),
                width=56,
                height=56,
                alignment=ft.alignment.center,
                bgcolor=tile_bg(tok),
                border_radius=8,
                border=ft.border.all(1, ft.Colors.WHITE24),
            )

        def rebuild_grid():
            grid_rows.clear()
            row_boxes.clear()
            tiles.clear()
            for r in range(ROWS):
                row_tiles = []
                for c in range(COLS):
                    cont = make_cell("", "")
                    tiles.append((cont.content, cont))
                    row_tiles.append(cont)

                row_box = ft.Container(
                    content=ft.Row(row_tiles, alignment=ft.MainAxisAlignment.CENTER, spacing=6),
                    offset=ft.Offset(0, 0),
                    animate_offset=ft.Animation(120, "easeInOut"),
                )
                row_boxes.append(row_box)
                grid_rows.append(row_box)

        rebuild_grid()

        status_text = ft.Text("Gõ chữ cái. Enter để submit và Backspace để xóa", size=14)
        hint_text = ft.Text("", size=14, opacity=0.9)

        def kb_color(ch: str):
            tok = game.kb_state.get(ch, "")
            return tile_bg(tok)

        def press_key(ch: str):
            if ch == "enter":
                on_enter()
                return
            if ch == "del":
                on_backspace()
                return
            if game.row >= ROWS:
                return
            if game.col < COLS:
                game.board[game.row][game.col] = ch
                game.col += 1
                refresh_board()

        def on_backspace():
            if game.row >= ROWS:
                return
            if game.col > 0:
                game.col -= 1
                game.board[game.row][game.col] = ""
                refresh_board()

        def invalidate(msg: str):
            status_text.value = msg
            page.update()

        async def shake_current_row():
            idx = min(max(game.row, 0), ROWS - 1)
            if idx >= len(row_boxes):
                return
            rb = row_boxes[idx]
            rb.offset = ft.Offset(0.06, 0)
            page.update()
            await asyncio.sleep(0.10)
            rb.offset = ft.Offset(-0.06, 0)
            page.update()
            await asyncio.sleep(0.08)
            rb.offset = ft.Offset(0, 0)
            if hasattr(page, "update_async"):
                await page.update_async()
            else:
                page.update()

        def show_answer_row():
            ans_tiles = []
            for i in range(COLS):
                ch = game.answer[i].upper()
                ans_tiles.append(
                    ft.Container(
                        content=ft.Text(ch, weight=ft.FontWeight.BOLD, size=24),
                        width=56,
                        height=56,
                        alignment=ft.alignment.center,
                        bgcolor=ft.Colors.GREEN,
                        border_radius=8,
                        border=ft.border.all(1, ft.Colors.WHITE24),
                    )
                )
            answer_row = ft.Container(
                content=ft.Row(ans_tiles, alignment=ft.MainAxisAlignment.CENTER, spacing=6)
            )
            grid_rows.append(answer_row)
            grid_box.content = ft.Column(grid_rows, spacing=6, alignment=ft.MainAxisAlignment.CENTER)
            page.update()

        def on_enter():
            if game.row < ROWS and game.col < COLS:
                invalidate("5 chữ cái")
                page.run_task(shake_current_row)
                return

            if game.row >= ROWS:
                invalidate("Bạn đã thua rồi. Nhấn New Game để chơi lại")
                return

            guess = game.current_guess()

            if guess not in ALLOWED:
                invalidate("Từ không hợp lệ")
                page.run_task(shake_current_row)
                return

            fb = score_guess(game.answer, guess)
            game.apply_feedback(guess, fb)
            game.col = 0
            refresh_board()
            color_keyboard()

            if guess == game.answer:
                status_text.value = f"Đáp án: {game.answer}. Nhấn New Game để chơi lại"
                hint_enabled["v"] = False
                hint_btn.opacity = 0.5
                page.update()
                game.row = ROWS
                return

            game.row += 1

            if game.row == ROWS:
                status_text.value = f"Bạn đã hết lượt. Đáp án: {game.answer}"
                hint_enabled["v"] = False
                hint_btn.opacity = 0.5
                page.update()
                show_answer_row()

        def refresh_board():
            idx = 0
            for r in range(ROWS):
                for c in range(COLS):
                    t_ctrl, cont = tiles[idx]
                    t_ctrl.value = game.board[r][c].upper()
                    cont.bgcolor = tile_bg(game.state[r][c])
                    idx += 1
            page.update()

        def color_keyboard():
            for tile in kb_tiles:
                key_val = tile.data
                if key_val and len(key_val) == 1:
                    tile.bgcolor = kb_color(key_val)
            page.update()

        def new_game(_: ft.ControlEvent = None):
            game.reset()
            status_text.value = "Chúc May Mắn!"
            hint_text.value = ""
            hint_btn.disabled = False
            rebuild_grid()
            grid_box.content = ft.Column(grid_rows, spacing=6, alignment=ft.MainAxisAlignment.CENTER)
            refresh_board()
            color_keyboard()
            page.update()
            hint_enabled["v"] = True
            hint_btn.opacity = 1.0
            page.update()

        kb_layout = [
            list("qwertyuiop"),
            list("asdfghjkl"),
            ["enter"] + list("zxcvbnm") + ["del"],
        ]

        kb_tiles = []
        kb_rows_ui = []

        def key_tile(label_text: str, key_value: str):
            bg = kb_color(key_value) if len(key_value) == 1 else ft.Colors.GREY_800
            w = 42 if len(key_value) == 1 else 78
            t = ft.Text(label_text, size=12, weight=ft.FontWeight.W_600)
            tile = ft.Container(
                data=key_value,
                content=t,
                width=w,
                height=40,
                alignment=ft.alignment.center,
                bgcolor=bg,
                border_radius=6,
                border=ft.border.all(1, ft.Colors.WHITE24),
                on_click=lambda e: press_key(e.control.data),
            )
            return tile

        kb_rows_ui.clear()
        kb_tiles.clear()
        for row in kb_layout:
            row_controls = []
            for key in row:
                label = key.upper() if len(key) == 1 else ("ENTER" if key == "enter" else "DEL")
                tile = key_tile(label, key)
                kb_tiles.append(tile)
                row_controls.append(tile)
            kb_rows_ui.append(
                ft.Row(row_controls, alignment=ft.MainAxisAlignment.CENTER, spacing=6)
            )

        grid_box = ft.Container(
            content=ft.Column(grid_rows, spacing=6, alignment=ft.MainAxisAlignment.CENTER)
        )

        def green_positions():
            greens = set()
            for r in range(ROWS):
                for c in range(COLS):
                    if game.state[r][c] == "g":
                        greens.add(c)
            return greens

        def make_hint_pattern():
            import random as _rnd
            greens = green_positions()
            candidates = [i for i in range(COLS) if i not in greens]
            idx = _rnd.choice(candidates) if candidates else _rnd.randrange(COLS)
            pattern = "".join(game.answer[i].upper() if i == idx else "⬛" for i in range(COLS))
            return f"Hint: {pattern}"

        def show_hint(_: ft.ControlEvent = None):
            if game.hint_used:
                invalidate("Bạn đã sử dụng gợi ý")
                return
            game.hint_used = True
            hint_text.value = make_hint_pattern()
            hint_enabled["v"] = False
            hint_btn.opacity = 0.5
            page.update()

        def on_key(e: ft.KeyboardEvent):
            k = e.key
            if k in ("Enter", "NumpadEnter"):
                on_enter()
            elif k in ("Backspace", "Delete"):
                on_backspace()
            else:
                ch = (k or "").lower()
                if len(ch) == 1 and ch.isalpha():
                    press_key(ch)

        page.on_keyboard_event = on_key

        def focus_board():
            try:
                page.set_focus(grid_box)
            except:
                pass

        def ghost_btn(label_text: str, on_click, tooltip: str = "", width: int = 78, height: int = 40):
            lbl_ref = ft.Ref[ft.Text]()
            base = ft.Container(
                content=ft.Text(label_text, ref=lbl_ref, size=12, weight=ft.FontWeight.W_600),
                width=width,
                height=height,
                alignment=ft.alignment.center,
                bgcolor=ft.Colors.WHITE10,
                border=ft.border.all(1, ft.Colors.WHITE24),
                border_radius=6,
                tooltip=tooltip,
                ink=False,
            )
            hit = ft.Container(
                width=width,
                height=height,
                bgcolor=ft.Colors.TRANSPARENT,
                on_click=lambda e: (on_click(e), focus_board()),
            )
            return ft.Stack([base, hit]), lbl_ref

        is_muted = {"v": False}

        def toggle_sound(e):
            if not music_ref.current:
                return
            if is_muted["v"]:
                music_ref.current.volume = 0.5
                sound_text_ref.current.value = "🔊"
                is_muted["v"] = False
            else:
                music_ref.current.volume = 0
                sound_text_ref.current.value = "🔇"
                is_muted["v"] = True
            page.update()

        hint_enabled = {"v": True}

        def on_hint_click(e):
            if not hint_enabled["v"]:
                return
            show_hint(e)
            hint_enabled["v"] = False
            hint_btn.opacity = 0.5
            page.update()

        def back_to_menu(_):
            stop_music()
            page.on_keyboard_event = None
            show_menu()

        back_btn, _           = ghost_btn("← Menu", back_to_menu, "Về menu", width=72)
        sound_btn, sound_text_ref = ghost_btn("🔊", toggle_sound, "Bật/Tắt nhạc", width=42)
        hint_btn,  hint_text_ref  = ghost_btn("Hint", lambda e: on_hint_click(e), "Hiển thị gợi ý")
        new_btn,   _              = ghost_btn("New Game", new_game, "Chơi lại")

        header = ft.Row([
            back_btn,
            ft.Text("Wordle", size=22, weight=ft.FontWeight.W_700),
            ft.Container(expand=True),
            sound_btn,
            hint_btn,
            new_btn,
        ])

        def loop_music(e):
            if e.data == "completed" and music_ref.current:
                music_ref.current.play()

        music = fta.Audio(
            ref=music_ref,
            src="background.mp3",
            autoplay=True,
            volume=0.8,
        )
        music.on_state_changed = loop_music
        if music not in page.overlay:
            page.overlay.append(music)

        game_view = ft.Column(
            [
                header,
                grid_box,
                ft.Divider(opacity=0),
                status_text,
                hint_text,
                ft.Divider(opacity=0),
                *kb_rows_ui,
                ft.Text(f"Số lượng từ: {len(ANSWERS)}", size=12, opacity=0.6),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
            spacing=12,
        )

        body.content = game_view
        page.update()

    def run_special_game(e=None):
        stop_music()
        page.run_task(_run_special_game_async)

    async def _run_special_game_async():
        test_path = ROOT / "test.py"
        if not test_path.exists():
            page.snack_bar = ft.SnackBar(ft.Text("thiếu file"))
            page.snack_bar.open = True
            page.update()
            return

        try:
            stop_music()
            page.window.minimized = True
            page.update()

            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(test_path)
            )
            await proc.wait()

        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Lỗi chạy: {ex}"))
            page.snack_bar.open = True
            page.update()
        finally:
            page.window.minimized = False
            page.clean()
            page.add(body)
            show_menu()
            page.update()

    page.add(body)
    show_menu()
    
if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.FLET_APP, assets_dir="assets")