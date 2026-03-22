import flet as ft
import queue
import threading
import time
import math

class FletUI:
    def __init__(self, state_queue):
        self.state_queue = state_queue
        self.page = None
        self.status_icon = None
        self.status_text = None
        self.transcription_text = None
        self.response_text = None
        self.emotion_text = None
        self.main_container = None
        self.is_running = True
        self._animation_tick = 0

    def run(self):
        ft.app(target=self.main)

    def main(self, page: ft.Page):
        self.page = page
        page.title = "小德 (Xiaode)"
        page.window_width = 420
        page.window_height = 650
        page.window_always_on_top = True
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 0
        page.spacing = 0
        page.window_bgcolor = ft.Colors.TRANSPARENT
        page.bgcolor = ft.Colors.TRANSPARENT
        
        # 渐变配色方案
        self.colors = {
            "idle": ["#1A1A1A", "#2D2D2D"],
            "listening": ["#1E3A8A", "#1E40AF"], # 蓝色
            "thinking": ["#4C1D95", "#5B21B6"],  # 紫色
            "speaking": ["#064E3B", "#065F46"],  # 绿色
            "action": ["#78350F", "#92400E"],    # 橙色
            "locked": ["#450A0A", "#7F1D1D"],    # 红色
        }

        # 状态图标容器 (带动画)
        self.status_icon = ft.Text("💤", size=100)
        self.icon_container = ft.Container(
            content=self.status_icon,
            alignment=ft.Alignment.CENTER,
            animate_scale=ft.Animation(600, ft.AnimationCurve.ELASTIC_OUT),
        )
        
        self.status_text = ft.Text("空闲中", size=22, weight=ft.FontWeight.W_600, color="white70")
        self.emotion_text = ft.Text("", size=14, color="pink200", italic=True)

        # 文本内容区
        self.transcription_text = ft.Text("", size=16, color="white60", italic=True)
        self.response_text = ft.Text("", size=18, color="white", weight=ft.FontWeight.W_400)

        # 玻璃拟态主容器
        self.main_container = ft.Container(
            content=ft.Column(
                [
                    ft.Divider(height=40, color="transparent"),
                    ft.Row([self.icon_container], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Row([self.status_text], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Row([self.emotion_text], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Divider(height=30, color="white24"),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("YOU", size=12, weight=ft.FontWeight.BOLD, color="white30"),
                            self.transcription_text,
                        ], spacing=5),
                        padding=ft.padding.only(left=10)
                    ),
                    ft.Divider(height=20, color="transparent"),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("XIAODE", size=12, weight=ft.FontWeight.BOLD, color="pink300"),
                            self.response_text,
                        ], spacing=8),
                        padding=ft.padding.only(left=10)
                    ),
                ],
                scroll=ft.ScrollMode.AUTO,
                horizontal_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=30,
            expand=True,
            border_radius=30,
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment.BOTTOM_RIGHT,
                colors=self.colors["idle"],
            ),
            blur=ft.Blur(20, 20),
            border=ft.border.all(1, "white10"),
            shadow=ft.BoxShadow(
                blur_radius=50,
                color="black54",
                offset=ft.Offset(0, 20),
            )
        )

        page.add(self.main_container)

        # 启动监听和动画线程
        threading.Thread(target=self.listen_for_updates, daemon=True).start()
        threading.Thread(target=self.animation_loop, daemon=True).start()

    def animation_loop(self):
        """处理微动画，如呼吸效果"""
        while self.is_running:
            if self.page:
                self._animation_tick += 0.1
                # 倾听时通过 scale 模拟呼吸
                if self.status_icon.value == "🎙️":
                    scale = 1.0 + 0.1 * math.sin(self._animation_tick * 2)
                    self.icon_container.scale = scale
                    try:
                        self.page.update()
                    except: pass
            time.sleep(0.05)

    def listen_for_updates(self):
        while self.is_running:
            try:
                state = self.state_queue.get(timeout=0.1)
                if "status" in state:
                    self.update_ui_status(state["status"])
                if "transcription" in state:
                    self.transcription_text.value = state["transcription"]
                if "response" in state:
                    self.response_text.value = state["response"]
                if "emotion" in state:
                    self.emotion_text.value = f"当前的你：{state['emotion']}" if state['emotion'] else ""
                
                self.page.update()
            except queue.Empty:
                continue
            except Exception as e:
                continue

    def update_ui_status(self, status):
        self.status_icon.value = status
        status_map = {
            "💤": ("空闲中", "idle"),
            "🎙️": ("正在听...", "listening"),
            "🤔": ("思考中...", "thinking"),
            "🗣️": ("正在说话...", "speaking"),
            "⚙️": ("执行中...", "action"),
            "🔒": ("已锁定", "locked"),
            "🌙": ("休眠中", "idle"),
            "👂": ("追问中...", "listening")
        }
        
        text, color_key = status_map.get(status, ("工作中", "idle"))
        self.status_text.value = text
        self.main_container.gradient.colors = self.colors.get(color_key, self.colors["idle"])
        
        # 状态切换时的弹性动画
        self.icon_container.scale = 1.2
        time.sleep(0.1)
        self.icon_container.scale = 1.0

def start_flet_ui(state_queue):
    ui = FletUI(state_queue)
    ui.run()

def start_flet_ui(state_queue):
    ui = FletUI(state_queue)
    ui.run()
