# https://github.com/official-pikafish/Pikafish
# https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition/tree/main

import cv2
import os
import time
import subprocess
import platform
import numpy as np
from pathlib import Path
from core.chessboard_detector import ChessboardDetector

# --- 配置区 ---
CURRENT_DIR = Path(__file__).parent
ENGINE_PATH = str(CURRENT_DIR / "pikafish-sse41-popcnt.exe")
if platform.system() == "Linux":
    ENGINE_PATH = "./pikafish-sse41-popcnt"

# 搜索深度
SEARCH_DEPTH = 18

TEMP_IMAGE_PATH = "adb_screenshot.png"

detector = ChessboardDetector(
    pose_model_path="onnx/pose/4_v6-0301.onnx",
    full_classifier_model_path="onnx/layout_recognition/nano_v3-0319.onnx"
)


def order_points(pts):
    """
    对检测到的4个角点进行排序，顺序为
    Index 0: 左上
    Index 1: 右上
    Index 2: 右下
    Index 3: 左下
    """
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
    
    # 定义一个标准虚拟棋盘坐标系 (宽 800, 高 900)
    # 无论屏幕如何，总是以标准的红方在下的视角来建立虚拟坐标
    src_pts = np.array([
        [0, 0],      # 对应左上角 (a9)
        [800, 0],    # 对应右上角 (i9)
        [800, 900],  # 对应右下角 (i0)
        [0, 900]     # 对应左下角 (a0)
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    col = ord(move_str[0]) - ord('a')   # a-i -> 0-8
    row = 9 - int(move_str[1])          # 9-0 -> 0-9
    
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
    if len(move) != 4:
        print(f"着法格式无法识别: {move}")
        return False
        
    start_pos = move[:2]
    end_pos = move[2:]
    
    try:
        start_x, start_y = get_pixel_coords(start_pos, keypoints, side_to_move)
        end_x, end_y = get_pixel_coords(end_pos, keypoints, side_to_move)
        
        print(f" -> [触控] 点击起手棋子: {start_pos} -> 坐标: ({start_x}, {start_y})")
        subprocess.run(["adb", "shell", "input", "tap", str(start_x), str(start_y)], check=True)
        
        time.sleep(0.02)
        
        print(f" -> [触控] 点击目标落点: {end_pos} -> 坐标: ({end_x}, {end_y})")
        subprocess.run(["adb", "shell", "input", "tap", str(end_x), str(end_y)], check=True)
        return True
    except Exception as e:
        print(f"ADB 触控执行失败: {e}")
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


def get_best_move(fen):
    if not os.path.exists(ENGINE_PATH):
        return f"错误: 找不到Pikafish {ENGINE_PATH}", None

    process = subprocess.Popen(
        ENGINE_PATH,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8'
    )

    try:
        process.stdin.write("uci\n")
        process.stdin.write("isready\n")
        process.stdin.write(f"position fen {fen}\n")
        process.stdin.write(f"go depth {SEARCH_DEPTH}\n")
        process.stdin.flush()

        best_move = ""
        score = ""

        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()
            
            if "info" in line and "score cp" in line:
                parts = line.split()
                try:
                    score_idx = parts.index("cp") + 1
                    score = parts[score_idx]
                except:
                    pass
            
            if line.startswith("bestmove"):
                best_move = line.split()[1]
                break
        
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.terminate()

        return best_move, score

    except Exception as e:
        return f"分析出错: {e}", None


def capture_screen_via_adb(output_path):
    try:
        subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/chess_screen.png"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["adb", "pull", "/sdcard/chess_screen.png", output_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print("adb 命令执行失败")
        return False
    except Exception as e:
        print(f"截图过程中出现未知错误: {e}")
        return False


def process_and_analyze(image_path, side_to_move):
    if not os.path.exists(image_path):
        print(f"错误: 找不到截图文件 {image_path}")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("图片读取失败")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        print("1. 识别棋盘布局与定位...")
        keypoints, _ = detector.pred_keypoints(img_bgr)
        
        res = detector.pred_detect_board_and_classifier(img_rgb)
        _, _, cells_labels_str, _, time_info = res
        
        fen = board_to_fen(cells_labels_str, side_to_move=side_to_move)
        print(f"\n[生成 FEN 码]: {fen}")

        print("\n2. 启动 Pikafish ...")
        move, score = get_best_move(fen)

        print("\n" + "="*40)
        print(f"【分析】")
        print(f"当前走子方: {'红方' if side_to_move == 'w' else '黑方'}")
        print(f"最佳着法: {move}")
        if score:
            print(f"局势评估: {score}")
        print("="*40)
        print(f"识别耗时: {time_info}\n")


        if move and not move.startswith("错误") and len(move) == 4:
            if keypoints is not None and len(keypoints) >= 4:
                print("3. 执行 adb 自动触控操作...")
                execute_adb_move(move, keypoints, side_to_move)
            else:
                print("无法自动落子：未能识别到足够的棋盘关键点坐标")
        else:
            print("未能获取有效的分析着法，跳过落子操作")

    except Exception as e:
        print(f"分析处理失败: {e}")


if __name__ == "__main__":
    print("=== 中国象棋 自动落子工具 ===")
    
    side_to_move = ""
    while side_to_move not in ['w', 'b']:
        user_input = input("请设置当前下子方 (输入 w 代表红方，输入 b 代表黑方): ").strip().lower()
        if user_input in ['w', 'b']:
            side_to_move = user_input
        else:
            print("输入格式不正确，请重新输入")

    print(f"\n已配置初始下子方为: {'红方(w)' if side_to_move == 'w' else '黑方(b)'}")
    print("准备就绪：")
    print("  - [回车/Enter] 键：通过 ADB 截图并开始分析落子")
    print("  - 输入 'c'：切换当前下子方")
    print("  - 输入 'q'：退出程序")

    while True:
        current_side_str = "红方(w)" if side_to_move == "w" else "黑方(b)"
        command = input(f"[{current_side_str}] 请输入指令: ").strip().lower()

        if command == 'q':
            break
        elif command == 'c':
            side_to_move = 'b' if side_to_move == 'w' else 'w'
            print(f"切换当前下子方为: {'红方(w)' if side_to_move == 'w' else '黑方(b)'}")
            continue
        elif command == '':
            if capture_screen_via_adb(TEMP_IMAGE_PATH):
                process_and_analyze(TEMP_IMAGE_PATH, side_to_move)
        else:
            print("无法识别的指令")