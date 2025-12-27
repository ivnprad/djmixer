from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog,QMessageBox
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel

import asyncio
from Core.PlaySongs import PlaySongsAlt
import threading
import queue

def RunAsyncInThread(resume=None,stopEvent=None):
    asyncio.run(PlaySongsAlt(resume,stopEvent))

class AudioPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.data_queue = queue.Queue()

    def init_ui(self):
        self.load_button = QPushButton('Load', self)
        self.play_button = QPushButton('Play', self)
        self.pause_button = QPushButton('Pause', self)
        self.stop_button = QPushButton('Stop', self)
        self.label = QLabel("Waiting for data...")

        self.load_button.clicked.connect(self.load_file)
        self.play_button.clicked.connect(self.play_audio)
        self.pause_button.clicked.connect(self.pause_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        layout = QVBoxLayout(self)
        layout.addWidget(self.load_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.label)

        self.setWindowTitle('Simple Audio Player')
        self.setGeometry(300, 300, 300, 200)

        self.stopPlayer = None
        self.playThread = None
        self.resume=None

        # Poll the queue every 100ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_queue)
        self.timer.start(100)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "MP3 files (*.mp3);;All files (*)")

    def play_audio(self):
        self.stopPlayer = asyncio.Event()
        self.playThread = threading.Thread(target=RunAsyncInThread, args=(self.resume,self.stopPlayer))
        self.playThread.start()
        print("playing...")

    def pause_audio(self):
        print("pausing...")
        if self.stopPlayer!=None:
            self.stopPlayer.set()
        if self.playThread!=None:
            self.playThread.join()
            self.playThread = None
        self.resume=True
        print("paused")

    def stop_audio(self):
        print("stopping...")
        if self.stopPlayer!=None:
            self.stopPlayer.set()
        if self.playThread!=None:
            self.playThread.join()
            self.playThread = None
        self.resume=True
        print("stopped")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.stop_audio()
            event.accept()
        else:
            event.ignore()  

    def poll_queue(self):
        try:
            while True:
                data = self.data_queue.get_nowait()
                self.label.setText(data)
        except queue.Empty:
            pass

if __name__ == '__main__':
    app = QApplication([])
    ex = AudioPlayer()
    ex.show()
    app.exec()
