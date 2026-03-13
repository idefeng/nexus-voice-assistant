import flet as ft
from flet import colors, icons, FontWeight, ThemeMode
import queue
import threading
import time

class FletUI:
    def __init__(self, state_queue):
        self.state_queue = state_queue
        self.page = None
        self.status_icon = None
        self.status_text = None
        self.transcription_text = None
        self.response_text = None
        self.is_running = True

    def run(self):
        ft.app(target=self.main)

    def main(self, page: ft.Page):
        self.page = page
        page.title = "小德 (Xiaode)"
        page.window_width = 400
        page.window_height = 600
        page.window_always_on_top = True
        page.theme_mode = ThemeMode.DARK
        page.bgcolor = colors.TRANSPARENT
        page.window_bgcolor = colors.TRANSPARENT
        page.window_title_bar_hidden = False # 开发阶段保留标题栏
        
        # 状态图标 (大大的 Emoji)
        self.status_icon = ft.Text("💤", size=100)
        self.status_text = ft.Text("空闲中", size=20, weight=FontWeight.BOLD)
        
        # 实时识别文本
        self.transcription_text = ft.Text("", size=16, color=colors.GREY_400, italic=True)
        
        # AI 回复文本
        self.response_text = ft.Text("", size=18, color=colors.WHITE)

        # 布局
        page.add(
            ft.Container(
                content=ft.Column(
                    [
                        ft.Divider(height=20, color=colors.TRANSPARENT),
                        ft.Row([self.status_icon], alignment=ft.MainAxisAlignment.CENTER),
                        ft.Row([self.status_text], alignment=ft.MainAxisAlignment.CENTER),
                        ft.Divider(height=20),
                        ft.Column([
                            ft.Text("你:", size=14, color=colors.GREY_500),
                            self.transcription_text,
                        ]),
                        ft.Divider(height=10, color=colors.TRANSPARENT),
                        ft.Column([
                            ft.Text("小德:", size=14, color=colors.GREY_500),
                            self.response_text,
                        ]),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=20,
                bgcolor=colors.with_opacity(0.85, colors.BLACK),
                border_radius=20,
                expand=True,
            )
        )

        # 启动监听线程
        threading.Thread(target=self.listen_for_updates, daemon=True).start()

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
                if "append_response" in state:
                    self.response_text.value += state["append_response"]
                
                self.page.update()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"UI Update Error: {e}")

    def update_ui_status(self, status):
        self.status_icon.value = status
        status_map = {
            "💤": "空闲中",
            "🎙️": "正在听...",
            "🤔": "思考中...",
            "🗣️": "正在说话...",
            "⚙️": "执行中...",
            "🔒": "已锁定",
            "🌙": "休眠中",
            "👂": "追问中..."
        }
        self.status_text.value = status_map.get(status, "工作中")

def start_flet_ui(state_queue):
    ui = FletUI(state_queue)
    ui.run()
