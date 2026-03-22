import queue
import threading

class UIManager:
    """统一 UI 管理器，支持不同 UI 后端的切换"""
    def __init__(self, ui_type="none"):
        self.ui_type = ui_type
        self.state_queue = queue.Queue()
        self.ui_instance = None
        self._title = "💤"

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.update_state({"status": value})

    def update_state(self, state_dict):
        """发送状态更新到 UI 进程/线程"""
        if self.ui_type != "none":
            self.state_queue.put(state_dict)

    def set_ui_instance(self, instance):
        self.ui_instance = instance

    def show_notification(self, title, subtitle, message):
        """显示通知"""
        if self.ui_type == "rumps" and self.ui_instance:
            self.ui_instance.notification(title, subtitle, message)
        # TODO: Flet notification implementation

ui_manager = None

def init_ui_manager(ui_type):
    global ui_manager
    ui_manager = UIManager(ui_type)
    return ui_manager
