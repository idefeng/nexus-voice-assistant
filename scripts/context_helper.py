import subprocess
import os

def get_active_window_info():
    """获取 macOS 当前最前端的应用名称"""
    ascript = 'tell application "System Events" to get name of first process whose frontmost is true'
    try:
        proc = subprocess.Popen(['osascript', '-e', ascript], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        if out:
            return out.decode('utf-8').strip()
    except:
        pass
    return "Unknown"

def get_system_load():
    """获取简单的系统负载信息"""
    try:
        load = os.getloadavg()[0]
        return f"{load:.2f}"
    except:
        return "N/A"

if __name__ == "__main__":
    print(f"Active App: {get_active_window_info()}")
    print(f"System Load: {get_system_load()}")
