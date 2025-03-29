import sys
import asyncio
import threading
import queue
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import QTimer

from dataclasses import dataclass
from typing import Optional, Dict, Any
import uuid

command_q = queue.Queue()

@dataclass
class Command:
    command:str
    request_id:str
    data: Optional[Dict[str,Any]]=None
    response_queue: Optional[queue.Queue]=None # for getting a response

@dataclass
class Response:
    request_id:str
    status:str #'ok','error'
    data:Optional[Any]=None
    error:Optional[str]=None


def HandleCommand(cmd:Command):
    try:   
        if cmd.command=="play":
            song_id = cmd.data["song_id"]
            #play song_id
            result = f"Playing {song_id}"
        elif cmd.command=="stop":
            #stopped
            result = "Stopped playback"
        elif cmd.command=="current_song":
            result = {"title": "Never Gonna Give You Up"}  # example stub
        elif cmd.command == "get_songs":
            result = ["Song A", "Song B", "Song C"]
        elif cmd.command == "add_song":
            path = cmd.data["song_path"]
            result = f"Added song at {path}"
        else:
            raise ValueError(f"Unknown command: {cmd.command}")

        if cmd.response_queue:
            response = Response(
                request_id=cmd.request_id,
                status='ok',
                data=result)
            cmd.response_queue.put(response)
        
    except Exception as e:
        if cmd.response_queue:
            response = Response(
                request_id=cmd.request_id,
                status='error',
                data=None,
                error=str(e))
            cmd.response_queue.put(response)

def SendCommand(command_queue:queue.Queue,command: str, data: Optional[dict]=None):
    req_id = str(uuid.uuid4())
    response_q = queue.Queue()

    cmd = Command(command=command,
                  request_id=req_id,
                  data=data,
                  response_queue=response_q
                  )
    command_queue.put(cmd)

    # We are gonna poll response in qtimer gui
    # Wait for response (could use a timeout here)
    #response:Response = response_q.get()
    #return response


# Create two queues for bi-directional communication
gui_to_worker = queue.Queue()
worker_to_gui = queue.Queue()

# --------------------------
# ASYNC WORKER FUNCTION
# --------------------------
async def async_worker(cmd_queue, result_queue):
    while True:
        try:
            cmd = cmd_queue.get_nowait()
            print(f"[Worker] Received command: {cmd}")

            if cmd["action"] == "greet":
                name = cmd.get("name", "stranger")
                await asyncio.sleep(1)  # simulate async work
                result_queue.put({"event": "greeting", "message": f"Hello, {name}!"})

            elif cmd["action"] == "long_task":
                await asyncio.sleep(3)
                result_queue.put({"event": "status", "message": "Long task finished."})

        except queue.Empty:
            await asyncio.sleep(0.1)  # prevent tight loop when idle

def thread_main():
    asyncio.run(async_worker(gui_to_worker, worker_to_gui))

async def ServerLoop(command_queue: queue.Queue):
    cnt=0
    while True:
        cmd: Command = command_queue.get()
        HandleCommand(cmd)
        await asyncio.sleep(1)
        cnt+=1
        if cnt==120:
            break

def ThreadServerLoop(command_queue):
    asyncio.run(ServerLoop(command_queue))

# --------------------------
# GUI CLASS
# --------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 + asyncio + thread messaging")

        self.label = QLabel("Press a button to send a command")
        self.button_greet = QPushButton("Greet")
        self.button_task = QPushButton("Start Long Task")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button_greet)
        layout.addWidget(self.button_task)
        self.setLayout(layout)

        self.button_greet.clicked.connect(self.send_greet)
        self.button_task.clicked.connect(self.send_long_task)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_worker_responses)
        self.timer.start(100)

    def send_greet(self):
        gui_to_worker.put({"action": "greet", "name": "You"})
        self.label.setText("Sent greeting command...")
        SendCommand(command_q,command="stop")
        # response:Response=SendCommand(command_q,command="stop")
        # if response.status=='ok':
        #     print("working")
        # else:
        #     print("not working")

    def send_long_task(self):
        gui_to_worker.put({"action": "long_task"})
        self.label.setText("Sent long task command...")

    def poll_worker_responses(self):
        try:
            while True:
                msg = worker_to_gui.get_nowait()
                print(f"[GUI] Received message: {msg}")

                if msg["event"] == "greeting":
                    self.label.setText(msg["message"])

                elif msg["event"] == "status":
                    self.label.setText(msg["message"])

                cmd= command_q.get_nowait()
                if cmd.response_queue:
                    print("response queue")
                    response:Response=cmd.response_queue.get_nowait()
                    print(response.data)
                # else:
                #     print("no response?")
        except queue.Empty:
            pass

# --------------------------
# MAIN ENTRY
# --------------------------
if __name__ == "__main__":
    # Start background worker thread
    # t = threading.Thread(target=thread_main, daemon=True)
    # t.start()

    #threading.Thread(target=server_loop, args=(command_q,), daemon=True).start()
    t = threading.Thread(target=ThreadServerLoop, args=(command_q,),daemon=True)
    t.start()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


# 🔍 What’s happening:
# GUI thread sends a command into gui_to_worker queue

# Background async_worker() listens for commands using get_nowait()

# It simulates some async work, then responds via worker_to_gui queue

# GUI polls that queue every 100ms using a QTimer and updates the UI

# This is a robust and extendable pattern — you can add more actions, events, parameters, and even file downloads, server calls, etc.

# Want to level this up with:

# cancelable tasks?

# progress updates?

# real-time logs? Just say the word 💬
