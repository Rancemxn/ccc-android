import struct
import socket
import subprocess
import threading
import time
import os
import numpy as np


class ScrcpyConnectionError(Exception):
    """自定义连接异常"""
    pass


class ScrcpyClient:
    """
    具有自动重连与高容错性的增强版 ScrcpyClient。
    """
    # ── 控制协议常量保持不变 ──
    TYPE_INJECT_KEYCODE = 0
    TYPE_INJECT_TEXT = 1
    TYPE_INJECT_TOUCH_EVENT = 2
    TYPE_INJECT_SCROLL_EVENT = 3

    ACTION_DOWN = 0
    ACTION_UP = 1

    CODEC_H264 = 0x68_32_36_34
    DISABLE_STREAM_EXPLICIT = 0
    DISABLE_STREAM_ERROR = 1

    DEFAULT_PORT = 27183
    SOCKET_NAME = "scrcpy"
    SERVER_JAR_DEVICE_PATH = "/data/local/tmp/scrcpy-server.jar"
    SERVER_VERSION = "4.0"

    def __init__(self, server_path="scrcpy-server-v4.0", max_size=1024,
                 video_bit_rate=2000000, log_level="warn", 
                 auto_reconnect=True, max_reconnect_attempts=5):
        """
        Args:
            server_path: 本地 scrcpy-server.jar 的路径
            max_size: 视频最大分辨率限制
            video_bit_rate: 比特率
            log_level: 服务端日志级别
            auto_reconnect: 是否开启自动重连
            max_reconnect_attempts: 最大连续重连尝试次数
        """
        self.server_path = os.path.abspath(server_path)
        self.max_size = max_size
        self.video_bit_rate = video_bit_rate
        self.log_level = log_level
        self.port = self.DEFAULT_PORT
        
        # 重连控制参数
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self._reconnecting_event = threading.Event()

        # 网络与进程句柄
        self.video_socket = None
        self.control_socket = None
        self.server_process = None
        self._ffmpeg_process = None

        # 视频参数元数据
        self.device_name = ""
        self.video_width = 0
        self.video_height = 0

        # 帧缓存及同步锁
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_available = threading.Event()

        # 状态标志与多线程安全锁
        self._connected = False
        self._running = False
        self._state_lock = threading.Lock()  # 保护连接状态转换的原子性

        # 线程句柄
        self._video_writer_thread = None
        self._ffmpeg_reader_thread = None
        self._reconnect_thread = None

    @property
    def connected(self):
        with self._state_lock:
            return self._connected

    def start(self):
        """
        启动客户端，支持并发调用保护。
        """
        with self._state_lock:
            if self._connected:
                return True
            self._running = True
            return self._establish_connection()

    def _establish_connection(self):
        """
        内部连接建立逻辑。执行前需确保已持有 _state_lock。
        """
        try:
            self._cleanup_resources()
            
            # 检测设备在线状态
            available, info = self.check_adb_device()
            if not available:
                raise ScrcpyConnectionError(f"未检测到可用的 ADB 设备: {info}")

            self._kill_existing_server()
            self._push_server()
            self._setup_forward()
            self._start_server()
            self._connect_sockets()
            self._parse_video_headers()
            
            self._start_video_decoder()
            self._connected = True
            self._frame_available.clear()
            print("[ScrcpyClient] 连接已成功建立。")
            return True
        except Exception as e:
            print(f"[ScrcpyClient] 建立连接过程中遇到错误: {e}")
            self._cleanup_resources()
            return False

    def _cleanup_resources(self):
        """
        释放所有网络和系统资源，设计为幂等操作，可在任意状态下安全调用。
        """
        # 1. 终止 FFmpeg 进程
        if self._ffmpeg_process:
            try:
                self._ffmpeg_process.stdin.close()
            except Exception:
                pass
            try:
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=1.0)
            except Exception:
                try:
                    self._ffmpeg_process.kill()
                except Exception:
                    pass
            self._ffmpeg_process = None

        # 2. 关闭 Socket 连接
        for sock in [self.video_socket, self.control_socket]:
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
        self.video_socket = None
        self.control_socket = None

        # 3. 终止 adb 端的 server 进程
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=1.0)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.server_process = None

        # 4. 清理残留帧状态，防止外部调用读取过期帧
        with self._frame_lock:
            self._latest_frame = None
        self._frame_available.clear()

    def _handle_disconnect(self):
        """
        当检测到数据接收/写入发生异常时，统一由此方法触发断开流程，并按需启动重连。
        """
        with self._state_lock:
            if not self._connected:
                return  # 已在断开处理中，避免重复触发
            
            print("[ScrcpyClient] 检测到连接非正常断开，正在清理旧资源...")
            self._connected = False
            self._cleanup_resources()

        if self.auto_reconnect and self._running:
            # 异步启动重连，避免阻塞当前的 I/O 线程
            if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
                self._reconnect_thread = threading.Thread(
                    target=self._reconnect_loop, 
                    name="ScrcpyReconnectThread",
                    daemon=True
                )
                self._reconnect_thread.start()

    def _reconnect_loop(self):
        """
        异步自动重连循环，采用指数退避策略。
        """
        if self._reconnecting_event.is_set():
            return
        self._reconnecting_event.set()

        attempt = 0
        backoff = 1.0  # 初始等待 1 秒
        
        print("[ScrcpyClient] 自动重连线程已启动...")
        while self._running:
            with self._state_lock:
                if self._connected:
                    break
            
            attempt += 1
            if self.max_reconnect_attempts > 0 and attempt > self.max_reconnect_attempts:
                print(f"[ScrcpyClient] 已达到最大重连次数限制 ({self.max_reconnect_attempts})，停止重连。")
                break

            print(f"[ScrcpyClient] 正在尝试重新连接 (第 {attempt} 次)...")
            
            with self._state_lock:
                success = self._establish_connection()
                if success:
                    break
            
            # 指数退避延时，最大等待 10 秒
            sleep_time = min(backoff * (1.5 ** (attempt - 1)), 10.0)
            time.sleep(sleep_time)

        self._reconnecting_event.clear()

    # ──────────────────────────────────────────────────────────
    #  增强防御性的 Public APIs
    # ──────────────────────────────────────────────────────────

    def screencap(self, timeout=5.0):
        """
        安全捕获当前图像。若处于重连或断线状态，不会返回历史脏帧。
        """
        if not self.connected:
            if timeout > 0:
                # 等待重连成功后新帧的到来
                signaled = self._frame_available.wait(timeout=timeout)
                if not signaled:
                    return None
            else:
                return None

        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def tap(self, x, y):
        """
        安全发送点击事件，避免底层 Socket 为空或写入异常导致程序崩溃。
        """
        try:
            self._send_touch_event(self.ACTION_DOWN, x, y, self.video_width, self.video_height, pressure=1.0)
            time.sleep(0.05)
            self._send_touch_event(self.ACTION_UP, x, y, self.video_width, self.video_height, pressure=0.0)
            return True
        except (ConnectionError, socket.error, AttributeError) as e:
            print(f"[ScrcpyClient] 点击事件发送失败 (已失去连接): {e}")
            self._handle_disconnect()
            return False

    def swipe(self, start_x, start_y, end_x, end_y, duration_ms=300):
        """
        安全发送滑动事件。
        """
        try:
            w, h = self.video_width, self.video_height
            steps = max(2, duration_ms // 16)
            delay = duration_ms / 1000.0 / steps

            self._send_touch_event(self.ACTION_DOWN, start_x, start_y, w, h, pressure=1.0)
            for i in range(1, steps):
                progress = i / steps
                x = int(start_x + (end_x - start_x) * progress)
                y = int(start_y + (end_y - start_y) * progress)
                self._send_touch_event(2, x, y, w, h, pressure=1.0)  # ACTION_MOVE = 2
                time.sleep(delay)
            self._send_touch_event(self.ACTION_UP, end_x, end_y, w, h, pressure=0.0)
            return True
        except (ConnectionError, socket.error, AttributeError) as e:
            print(f"[ScrcpyClient] 滑动事件发送失败 (已失去连接): {e}")
            self._handle_disconnect()
            return False

    def close(self):
        """
        主动关闭客户端，彻底释放资源并关闭重连线程。
        """
        print("[ScrcpyClient] 正在主动关闭连接...")
        self._running = False
        with self._state_lock:
            self._connected = False
            self._cleanup_resources()

    # ──────────────────────────────────────────────────────────
    #  健壮的底层通信封装 (Private)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _recv_exact(sock, n):
        """
        增加异常防护的套接字精准读取。
        """
        if sock is None:
            raise ConnectionError("读取失败：套接字为空")
        data = bytearray()
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
            except socket.error as e:
                raise ConnectionError(f"读取套接字时发生网络异常: {e}")
            
            if not chunk:
                raise ConnectionError("scrcpy-server 连接已断开 (收到 EOF)")
            data.extend(chunk)
        return bytes(data)

    def _send_touch_event(self, action, x, y, screen_width, screen_height,
                          pressure=1.0, pointer_id=0):
        """
        保护控制字发送，防止往已关闭的 socket 写入数据。
        """
        sock = self.control_socket
        if not self.connected or sock is None:
            raise ConnectionError("控制套接字未建立或已失效")

        pressure_raw = 0xFFFF if pressure >= 1.0 else (0x0000 if pressure <= 0.0 else int(pressure * 0x10000) & 0xFFFF)
        
        msg = struct.pack(
            '>BBqiiHHHii',
            self.TYPE_INJECT_TOUCH_EVENT,
            action,
            pointer_id,
            int(x),
            int(y),
            screen_width,
            screen_height,
            pressure_raw,
            0,
            0,
        )

        try:
            sock.sendall(msg)
        except socket.error as e:
            raise ConnectionError(f"无法发送控制指令: {e}")

    # ── 下列底层初始化方法在建立连接时被调用，若遇到错误会向上抛出异常由 _establish_connection 捕获 ──

    def _kill_existing_server(self):
        try:
            subprocess.run(
                ['adb', 'shell', 'pkill', '-f', 'com.genymobile.scrcpy.Server'],
                capture_output=True, timeout=3.0, stdin=subprocess.DEVNULL
            )
        except Exception:
            pass

    def _push_server(self):
        if not os.path.exists(self.server_path):
            raise FileNotFoundError(f"找不到 scrcpy-server 文件: {self.server_path}")
        result = subprocess.run(
            ['adb', 'push', self.server_path, self.SERVER_JAR_DEVICE_PATH],
            capture_output=True, text=True, timeout=10.0, stdin=subprocess.DEVNULL
        )
        if result.returncode != 0:
            raise RuntimeError(f"推送服务 jar 到设备失败: {result.stderr}")

    def _setup_forward(self):
        try:
            subprocess.run(
                ['adb', 'forward', '--remove', f'tcp:{self.port}'],
                capture_output=True, timeout=3.0, stdin=subprocess.DEVNULL
            )
        except Exception:
            pass
        result = subprocess.run(
            ['adb', 'forward', f'tcp:{self.port}', f'localabstract:{self.SOCKET_NAME}'],
            capture_output=True, text=True, timeout=5.0, stdin=subprocess.DEVNULL
        )
        if result.returncode != 0:
            raise RuntimeError(f"设置端口转发失败: {result.stderr}")

    def _start_server(self):
        shell_command = (
            f"export CLASSPATH={self.SERVER_JAR_DEVICE_PATH}; "
            f"app_process / com.genymobile.scrcpy.Server {self.SERVER_VERSION} "
            f"tunnel_forward=true audio=false control=true "
            f"send_device_meta=true send_frame_meta=true "
            f"send_dummy_byte=true send_stream_meta=true "
            f"video_codec=h264 log_level={self.log_level} "
            f"cleanup=false power_on=false stay_awake=false "
            f"show_touches=false clipboard_autosync=false max_fps=10"
        )
        if self.max_size > 0:
            shell_command += f" max_size={self.max_size}"
        if self.video_bit_rate != 8000000:
            shell_command += f" video_bit_rate={self.video_bit_rate}"

        self.server_process = subprocess.Popen(
            ['adb', 'shell', shell_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        
        # 异步读取输出，防止管道堵塞
        def silence_reader(pipe):
            try:
                for _ in iter(pipe.readline, b''):
                    pass
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        threading.Thread(target=silence_reader, args=(self.server_process.stdout,), daemon=True).start()
        threading.Thread(target=silence_reader, args=(self.server_process.stderr,), daemon=True).start()
        time.sleep(1.5)

    def _connect_sockets(self):
        self.video_socket = self._connect_with_retry()
        self.video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        time.sleep(0.2)
        self.control_socket = self._connect_with_retry()
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _connect_with_retry(self, max_retries=5, retry_delay=0.5):
        last_error = None
        for _ in range(max_retries):
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3.0)
                sock.connect(('127.0.0.1', self.port))
                return sock
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                last_error = e
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
                time.sleep(retry_delay)
        raise ConnectionError(f"TCP 连接失败: {last_error}")

    def _parse_video_headers(self):
        dummy = self._recv_exact(self.video_socket, 1)
        device_name_bytes = self._recv_exact(self.video_socket, 64)
        self.device_name = device_name_bytes.split(b'\x00')[0].decode('utf-8', errors='replace')

        codec_id_bytes = self._recv_exact(self.video_socket, 4)
        codec_id = struct.unpack('>I', codec_id_bytes)[0]
        if codec_id != self.CODEC_H264:
            raise RuntimeError(f"不支持的编码: 0x{codec_id:08x}")

        session_data = self._recv_exact(self.video_socket, 12)
        _, self.video_width, self.video_height = struct.unpack('>III', session_data)

    def _start_video_decoder(self):
        frame_size = self.video_width * self.video_height * 3
        self._ffmpeg_process = subprocess.Popen(
            [
                'ffmpeg', '-probesize', '32', '-analyzeduration', '0', '-flags', 'low_delay',
                '-f', 'h264', '-i', 'pipe:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-v', 'quiet', 'pipe:1'
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        def video_writer():
            try:
                while self._running:
                    # 读取 Scrcpy 帧头
                    header = self._recv_exact(self.video_socket, 12)
                    packet_size = struct.unpack('>I', header[8:12])[0]
                    if packet_size <= 0 or packet_size > (1 << 20):
                        continue

                    # 读取原始 H.264 数据
                    nal_data = self._recv_exact(self.video_socket, packet_size)
                    
                    # 写入解码器
                    self._ffmpeg_process.stdin.write(nal_data)
                    self._ffmpeg_process.stdin.flush()
            except (ConnectionError, OSError, AttributeError):
                # 捕获因连接中断、Socket关闭导致的管道破裂或读写异常
                self._handle_disconnect()
            finally:
                try:
                    self._ffmpeg_process.stdin.close()
                except Exception:
                    pass

        def ffmpeg_reader():
            try:
                while self._running:
                    frame_data = self._ffmpeg_process.stdout.read(frame_size)
                    if not frame_data or len(frame_data) != frame_size:
                        break
                    frame = np.frombuffer(frame_data, np.uint8).reshape(
                        (self.video_height, self.video_width, 3)
                    )
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                    self._frame_available.set()
            except Exception:
                pass
            finally:
                self._handle_disconnect()

        self._video_writer_thread = threading.Thread(target=video_writer, daemon=True)
        self._ffmpeg_reader_thread = threading.Thread(target=ffmpeg_reader, daemon=True)
        self._video_writer_thread.start()
        self._ffmpeg_reader_thread.start()

    @staticmethod
    def check_adb_device():
        try:
            result = subprocess.run(
                ['adb', 'devices'],
                capture_output=True, text=True, timeout=3.0, stdin=subprocess.DEVNULL
            )
            lines = result.stdout.strip().split('\n')
            device_lines = [l for l in lines[1:] if '\tdevice' in l]
            if device_lines:
                serial = device_lines[0].split('\t')[0]
                return True, serial
            return False, "无设备在线"
        except Exception as e:
            return False, str(e)