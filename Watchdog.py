import subprocess
from time import sleep
import psutil
import socket
import multiprocessing
from multiprocessing_logging import install_mp_handler
from Logging.Logmodule import SetupLogging
from Logging.WatchdogLogger import watchdogLogger
from FileHandling import DeleteFile


mainScript = "main.py"

def IsProcessRunning(targetScript):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] is None:
                continue
            if targetScript in ' '.join(proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  

    return False

# TODO add more verificaton steps here but for the moment .py
def CheckTargetScript(targetScript):
    if not (isinstance(targetScript, str) and targetScript.endswith(".py")):
        raise "target script should be a string ending with .py"  

def is_process_running(process):
    return process.poll() is None

def StartProcess(targetScript=mainScript, resume=None):
    command = ["python3", targetScript]
    if resume is not None:
        command += ["--resume", str(resume)]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def StopProcess(process):
    if process is not None:
        if process.pid is not None and process.poll() is None:
            process.terminate()
        sleep(2)
        if process.pid is not None and process.poll() is None:  
            process.kill()    

def watchdog():
    
    try:
        watchdogLogger.info("starting")
        aidj = mainScript
        CheckTargetScript(aidj)
        process=None

        aidjCounter = 0

        while True:
            aidjCounter += 1
            if IsProcessRunning(aidj):
                if aidjCounter >1:
                    aidjCounter=1
            else:
                resume = None
                if aidjCounter == 1:
                    watchdogLogger.debug(" clean up json files ")
                    DeleteFile("songsList.json")
                    DeleteFile("songsPlayed.json")
                    DeleteFile("AIDJ.log")
                if aidjCounter >1:
                    resume = "Yes"
                watchdogLogger.error(" AIDJ has stopped. Restarting ")
                process = StartProcess(aidj, resume)

            sleep(0.01)

    except KeyboardInterrupt:
        watchdogLogger.info("Program interrupted by user.")
        StopProcess(process)
    except Exception as e:
        watchdogLogger.error(f"An error occurred: {e}")
        StopProcess(process)

if __name__ == "__main__":
    watchdog()
