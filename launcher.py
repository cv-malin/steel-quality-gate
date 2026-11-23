"""launcher.py — exe 双击入口：自动选空闲端口、开浏览器、启动服务。"""
import socket
import sys
import threading
import webbrowser
import time

import server


def pick_port(start=8766, tries=30):
    for p in range(start, start + tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    return start


def main():
    port = pick_port(8766)
    sys.argv = [sys.argv[0], "--port", str(port)]
    url = f"http://127.0.0.1:{port}"

    def open_browser():
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    print(f"Steel Quality Gate -> {url}")
    server.main()


if __name__ == "__main__":
    main()
