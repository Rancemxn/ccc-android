# https://github.com/official-pikafish/Pikafish
# https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition/tree/main

import cv2
import os
import time
import subprocess
import threading
import platform
import multiprocessing
import numpy as np
import psutil
from pathlib import Path
from ppadb.client import Client as AdbClient
from core.chessboard_detector import ChessboardDetector

from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box
from rich.traceback import install

install(show_locals=True)

custom_theme = Theme({
    "info": "cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "move": "bold reverse green"
})
console = Console(theme=custom_theme)

CURRENT_DIR = Path(__file__).parent

ENGINE_PATH = str(CURRENT_DIR / "pikafish-sse41-popcnt.exe")
if platform.system() == "Linux":
    ENGINE_PATH = str(CURRENT_DIR / "pikafish-sse41-popcnt")

# 硬件资源分配
SYSTEM_CORES = multiprocessing.cpu_count()
ENGINE_THREADS = max(1, SYSTEM_CORES)
ENGINE_HASH_MB = 1024

# 默认每步最大思考时间（毫秒）
DEFAULT_THINK_TIME_MS = 3000

with console.status("[info]正在加载 ONNX 棋盘检测与布局识别模型...", spinner="dots"):
    detector = ChessboardDetector(
        pose_model_path="onnx/pose/4_v6-0301.onnx",
        full_classifier_model_path="onnx/layout_recognition/nano_v3-0319.onnx"
    )

def init_adb_device(host="127.0.0.1", port=5037):
    try:
        client = AdbClient(host=host, port=port)
        devices = client.devices()
        if not devices:
            console.print("[warning]警告: 未检测到任何已连接的 ADB 设备。请确保已开启 USB 调试，且 adb server 已启动。[/warning]")
            return None
        device = devices[0]
        console.print(f"[success]ADB 设备连接成功: {device.serial}[/success]")
        return device
    except Exception as e:
        console.print(f"[error]ADB 连接初始化失败: {e}[/error]")
        return None

adb_device = init_adb_device()

class PikafishEngine:
    def __init__(self, path, threads=4, hash_size=1024):
        self.path = path
        self.threads = threads
        self.hash_size = hash_size
        self.process = None

        self.uciok_received = threading.Event()
        self.ready_received = threading.Event()
        self.best_move_received = threading.Event()
        self.fen_received = threading.Event()
        
        self.latest_depth = 0
        self.latest_score = "N/A"
        self.best_move_val = None
        self.ponder_move_val = None
        self.captured_fen = None
        
        self.is_pondering = False
        self.ponder_target_fen = None
        self.last_best_move = None
        self.last_ponder_move = None
        
        self.start_engine()

    def start_engine(self):
        if not os.path.exists(self.path):
            console.print(f"[warning]警告: 找不到 Pikafish 引擎文件: {self.path}[/warning]")
            return False
        
        try:
            self.process = subprocess.Popen(
                self.path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding='utf-8'
            )
            
            try:
                p = psutil.Process(self.process.pid)
                if platform.system() == "Windows":
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(10)
            except Exception as e:
                console.print(f"[info][提示] 调整进程优先级失败: {e}[/info]")
            
            reader_thread = threading.Thread(
                target=self._reader_loop, 
                args=(self.process.stdout,), 
                daemon=True
            )
            reader_thread.start()

            self.uciok_received.clear()
            self._send("uci")
            self.uciok_received.wait(timeout=5)

            self._send(f"setoption name Threads value {self.threads}")
            self._send(f"setoption name Hash value {self.hash_size}")
            self._send("setoption name Ponder value true")  # 开启 Ponder

            self.is_ready()
            console.print(f"[success]Pikafish 引擎初始化成功 (线程数: {self.threads}, Hash: {self.hash_size}MB, 已调低优先级以保障运行流畅)[/success]")
            return True
        except Exception as e:
            console.print(f"[error]Pikafish 引擎启动失败: {e}[/error]")
            self.process = None
            return False

    def _reader_loop(self, stdout):
        while True:
            try:
                line = stdout.readline()
                if not line:
                    break
                line = line.strip()

                if "uciok" in line:
                    self.uciok_received.set()
                elif "readyok" in line:
                    self.ready_received.set()
                elif line.lower().startswith("fen:"):
                    self.captured_fen = line[4:].strip()
                    self.fen_received.set()
                elif "info" in line:
                    parts = line.split()
                    if "depth" in parts:
                        try:
                            self.latest_depth = int(parts[parts.index("depth") + 1])
                        except (ValueError, IndexError):
                            pass
                    if "score" in parts:
                        try:
                            idx = parts.index("score")
                            self.latest_score = f"{parts[idx+1]} {parts[idx+2]}"
                        except IndexError:
                            pass
                elif line.startswith("bestmove"):
                    parts = line.split()
                    self.best_move_val = parts[1] if len(parts) >= 2 else ""
                    if "ponder" in parts:
                        try:
                            ponder_idx = parts.index("ponder") + 1
                            self.ponder_move_val = parts[ponder_idx]
                        except (ValueError, IndexError):
                            self.ponder_move_val = None
                    else:
                        self.ponder_move_val = None
                    self.best_move_received.set()
            except Exception:
                break

    def _send(self, cmd):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(f"{cmd}\n")
                self.process.stdin.flush()
            except IOError:
                pass

    def is_ready(self):
        self.ready_received.clear()
        self._send("isready")
        self.ready_received.wait(timeout=5)

    def ensure_alive(self):
        if self.process is None or self.process.poll() is not None:
            console.print("[warning]检测到 Pikafish 进程意外终止，尝试重启中...[/warning]")
            self.is_pondering = False
            self.ponder_target_fen = None
            self.start_engine()

    def reset_game(self):
        self.ensure_alive()
        if self.is_pondering:
            self._send("stop")
            self._read_search_output()
            self.is_pondering = False
            self.ponder_target_fen = None
        self._send("ucinewgame")
        self.is_ready()
        console.print("[info][引擎] 已执行 ucinewgame，所有状态与缓存已安全清空。[/info]")

    def get_fen_after_moves(self, base_fen, moves_list):
        self.ensure_alive()
        self.fen_received.clear()
        self.ready_received.clear()
        
        moves_str = " ".join(moves_list)
        self._send(f"position fen {base_fen} moves {moves_str}")
        self._send("d")
        self._send("isready")
        
        if self.ready_received.wait(timeout=3):
            return self.captured_fen
        return None

    def _read_search_output(self, timeout=None):
        self.best_move_received.wait(timeout=timeout)
        best_move = self.best_move_val
        score_str = f"{self.latest_score} (计算深度: {self.latest_depth}层)" if self.latest_score != "N/A" else "N/A"
        ponder_move = self.ponder_move_val
        
        self.best_move_received.clear()
        return best_move, score_str, ponder_move

    def get_best_move(self, fen, movetime=1500):
        self.ensure_alive()
        if not self.process:
            return "错误: 引擎未正常运行", None

        try:
            fen = " ".join(fen.split())

            # Ponder 校验 命中逻辑
            if self.is_pondering and self.ponder_target_fen:
                actual_parts = fen.split()[:4]
                target_parts = self.ponder_target_fen.split()[:4]

                if actual_parts == target_parts:
                    console.print(f"\n[highlight][Ponder] ★ 命中预测！对手确实下了: {self.last_ponder_move}[/highlight]")
                    console.print(f"[info][Ponder] 启用后台缓存，继续思考 {movetime / 1000.0:.1f} 秒...[/info]")
                    
                    self.best_move_received.clear()
                    self._send("ponderhit")
                    time.sleep(movetime / 1000.0)
                    self._send("stop")
                    best_move, score_str, next_ponder_move = self._read_search_output()
                else:
                    console.print(f"\n[warning][Ponder] 预测未命中[/warning]")
                    console.print(f"[info][Ponder] 实际局面: {fen}[/info]")
                    console.print(f"[info][Ponder] 重置并启动常规搜索...[/info]")
                    
                    self.best_move_received.clear()
                    self._send("stop")
                    self._read_search_output()
                    
                    self.best_move_received.clear()
                    self.latest_depth = 0
                    self.latest_score = "N/A"
                    self._send(f"position fen {fen}")
                    self._send(f"go movetime {movetime}")
                    best_move, score_str, next_ponder_move = self._read_search_output()
            else:
                self.best_move_received.clear()
                self.latest_depth = 0
                self.latest_score = "N/A"
                self._send(f"position fen {fen}")
                self._send(f"go movetime {movetime}")
                best_move, score_str, next_ponder_move = self._read_search_output()

            self.last_best_move = best_move
            self.last_ponder_move = next_ponder_move

            if next_ponder_move:
                expected_fen = self.get_fen_after_moves(fen, [best_move, next_ponder_move])
                if expected_fen:
                    self.ponder_target_fen = " ".join(expected_fen.split())
                    self.is_pondering = True
                    
                    console.print(f"[info][Ponder] 预测对手下步为: {next_ponder_move}，正在启动后台异步思考...[/info]")
                    self._send(f"position fen {fen} moves {best_move} {next_ponder_move}")
                    self._send("go ponder")
                else:
                    self.is_pondering = False
                    self.ponder_target_fen = None
            else:
                self.is_pondering = False
                self.ponder_target_fen = None

            return best_move, score_str

        except Exception as e:
            self.is_pondering = False
            self.ponder_target_fen = None
            return f"搜索异常: {e}", None

    def close(self):
        if self.process:
            if self.is_pondering:
                self._send("stop")
            self._send("quit")
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.terminate()

engine = PikafishEngine(ENGINE_PATH, threads=ENGINE_THREADS, hash_size=ENGINE_HASH_MB)

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def get_pixel_coords(move_str, keypoints, side_to_move):
    if len(keypoints) < 4:
        raise ValueError("检测到的棋盘角点不足 4 个，无法计算投影")

    dst_pts = order_points(keypoints[:4])

    src_pts = np.array([
        [0, 0],
        [800, 0],
        [800, 900],
        [0, 900]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    col = ord(move_str[0]) - ord('a')
    row = 9 - int(move_str[1])

    virtual_x = col * 100
    virtual_y = row * 100

    if side_to_move == 'b':
        virtual_x = 800 - virtual_x
        virtual_y = 900 - virtual_y

    point = np.array([[[virtual_x, virtual_y]]], dtype="float32")
    transformed_point = cv2.perspectiveTransform(point, M)

    pixel_x, pixel_y = transformed_point[0][0]
    return int(pixel_x), int(pixel_y)

def execute_adb_move(move, keypoints, side_to_move):
    if adb_device is None:
        console.print("[error]错误: ADB 未连接，无法自动落子[/error]")
        return False

    if len(move) != 4:
        console.print(f"[warning]着法格式无法识别: {move}[/warning]")
        return False
        
    start_pos = move[:2]
    end_pos = move[2:]

    try:
        start_x, start_y = get_pixel_coords(start_pos, keypoints, side_to_move)
        end_x, end_y = get_pixel_coords(end_pos, keypoints, side_to_move)
        
        console.print(f" -> [info][触控] 点击起手棋子: {start_pos} -> 坐标: ({start_x}, {start_y})[/info]")
        adb_device.shell(f"input tap {start_x} {start_y}")
        
        console.print(f" -> [info][触控] 点击目标落点: {end_pos} -> 坐标: ({end_x}, {end_y})[/info]")
        adb_device.shell(f"input tap {end_x} {end_y}")
        return True
    except Exception as e:
        console.print(f"[error]ADB 触控执行失败: {e}[/error]")
        return False

def board_to_fen(cells_labels_str, side_to_move='w'):
    raw_rows = [r.strip() for r in cells_labels_str.strip().split('\n') if r.strip()]
    rows = [r.replace(" ", "") for r in raw_rows]

    fen_rows = []
    for row in rows:
        fen_row = ""
        empty_count = 0
        for char in row:
            if char in ['.', 'x']:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += char
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    return "/".join(fen_rows) + f" {side_to_move} - - 0 1"

def capture_screen_via_adb():
    if adb_device is None:
        console.print("[error]错误: ADB 设备未正常连接，无法截图[/error]")
        return None
    try:
        image_bytes = adb_device.screencap()
        if not image_bytes:
            console.print("[warning]未能获取到有效的截图数据[/warning]")
            return None

        img_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return img_bgr
    except Exception as e:
        console.print(f"[error]截图过程中出现异常: {e}[/error]")
        return None

def process_and_analyze(img_bgr, side_to_move, think_time_ms):
    if img_bgr is None:
        console.print("[error]错误: 传入的棋盘图像数据无效[/error]")
        return

    with console.status("[info]正在提取棋盘定位特征与分类棋位状态...[/info]", spinner="bouncingBar"):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            keypoints, _ = detector.pred_keypoints(img_bgr)
            res = detector.pred_detect_board_and_classifier(img_rgb)
            _, _, cells_labels_str, _, time_info = res
            fen = board_to_fen(cells_labels_str, side_to_move=side_to_move)
        except Exception as e:
            console.print(f"[error]识别模型运行失败: {e}[/error]")
            return

    console.print(f"\n[success]生成 FEN 码:[/success] [highlight]「{fen}」[/highlight]")

    with console.status(f"[info]正在调用 Pikafish 引擎分析 (最大分配时间: {think_time_ms / 1000:.1f} 秒)...[/info]", spinner="clock"):
        t0 = time.time()
        move, score = engine.get_best_move(fen, movetime=think_time_ms)
        engine_time = (time.time() - t0) * 1000

    table = Table(title="【局势分析】", box=box.DOUBLE_EDGE, show_header=False, title_style="bold magenta")
    table.add_row("当前下子方", "红方 (w)" if side_to_move == 'w' else "黑方 (b)", style="info")
    table.add_row("推荐最佳走法", f"[move] {move} [/move]")
    table.add_row("局势评估分", f"[highlight]{score}[/highlight]")
    table.add_row("引擎响应耗时", f"{engine_time:.1f} ms")
    table.add_row("神经网络识别耗时", f"{time_info}")
    console.print(table)

    if move and not move.startswith("错误") and len(move) == 4:
        if keypoints is not None and len(keypoints) >= 4:
            console.print("[info]3. 执行 adb 自动触控操作...[/info]")
            execute_adb_move(move, keypoints, side_to_move)
        else:
            console.print("[warning]无法自动落子：未能识别到足够的棋盘关键点坐标[/warning]")
    else:
        console.print("[warning]未能获取有效的分析着法，跳过落子操作[/warning]")


if __name__ == "__main__":
    console.print(Panel.fit(
        "=== 中国象棋 自动落子工具（Ponder & PPADB & Rich 优化版） ===",
        style="bold magenta",
        border_style="cyan"
    ))

    if adb_device is None:
        console.print("\n[warning]请检查 USB 连接和 ADB 服务是否启动（输入 'adb devices' 确认设备状态）[/warning]")
        console.print("[warning]程序已转为无 ADB 连接模式运行，届时将无法执行自动点击，仅显示分析结果。[/warning]\n")

    side_to_move = Prompt.ask("请设置当前下子方", choices=["w", "b"], default="w")

    default_think_time_ms = DEFAULT_THINK_TIME_MS
    time_input = Prompt.ask(f"请设置默认思考时间 (单位：秒，直接回车则默认 {DEFAULT_THINK_TIME_MS / 1000} 秒)", default="3.0")
    if time_input:
        try:
            default_think_time_ms = int(float(time_input) * 1000)
        except ValueError:
            console.print(f"[warning]输入格式有误，将使用默认思考时间 {DEFAULT_THINK_TIME_MS / 1000} 秒[/warning]")

    config_summary = (
        f"初始下子方: [highlight]{'红方(w)' if side_to_move == 'w' else '黑方(b)'}[/highlight]\n"
        f"全局默认思考时间: [highlight]{default_think_time_ms / 1000:.2f} 秒[/highlight]\n\n"
        "操作指南:\n"
        "  - [bold cyan]直接回车/Enter[/bold cyan]：通过 ADB 内存截屏并执行[bold yellow]默认时间[/bold yellow]分析落子\n"
        "  - [bold cyan]输入数字[/bold cyan] (如 3 或 1.5)：使用该[bold yellow]临时时间[/bold yellow]运行一次\n"
        "  - [bold cyan]输入 'c'[/bold cyan]：切换当前下子方\n"
        "  - [bold cyan]输入 'r'[/bold cyan]：重置对局缓存与 Ponder 状态\n"
        "  - [bold cyan]输入 'q'[/bold cyan]：退出程序"
    )
    console.print(Panel(config_summary, title="[success]配置就绪[/success]", border_style="green"))

    try:
        while True:
            current_side_str = "红方(w)" if side_to_move == "w" else "黑方(b)"
            default_seconds_str = f"{default_think_time_ms / 1000:.1f}s"
            
            command = console.input(f"\n[[bold info]{current_side_str}[/bold info] | 默认 {default_seconds_str}] 请输入指令 (回车/数字/c/r/q): ").strip().lower()

            if command == 'q':
                break
            elif command == 'c':
                side_to_move = 'b' if side_to_move == 'w' else 'w'
                console.print(f"[info]切换当前下子方为: {'红方(w)' if side_to_move == 'w' else '黑方(b)'}[/info]")
                continue
            elif command == 'r':
                engine.reset_game()
                continue
            elif command == '':
                img = capture_screen_via_adb()
                if img is not None:
                    process_and_analyze(img, side_to_move, default_think_time_ms)
            else:
                try:
                    temp_seconds = float(command)
                    if temp_seconds <= 0:
                        console.print("[warning]时间必须大于 0 秒，请重新输入[/warning]")
                        continue
                    
                    temp_think_time_ms = int(temp_seconds * 1000)
                    console.print(f"\n[info][临时调整] 本次分析将限时 {temp_seconds:.1f} 秒计算...[/info]")
                    
                    img = capture_screen_via_adb()
                    if img is not None:
                        process_and_analyze(img, side_to_move, temp_think_time_ms)
                        
                except ValueError:
                    console.print("[warning]无法识别的指令。请输入数字（代表临时时间）、直接回车、或快捷键 (c / r / q)[/warning]")
    finally:
        engine.close()
