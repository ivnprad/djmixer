import subprocess
import time
import psutil


def IsProcessRunning(targetScript):
    # Check if there is any running process that was started with the target script
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip processes with no command line information
            if proc.info['cmdline'] is None:
                continue

            # Check if the target script is in the command line arguments
            if targetScript in ' '.join(proc.info['cmdline']):
                return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Ignore any processes that no longer exist or can't be accessed
    return False

# TODO add more verificaton steps here but for the moment .py
def CheckTargetScript(targetScript):
    if not (isinstance(targetScript, str) and targetScript.endswith(".py")):
        raise "target script should be a string ending with .py"  

def is_process_running(process):
    return process.poll() is None

def StartProcess(targetScript):
    return subprocess.Popen(["python3", "aidj.py"])

def StopProcess(process):
    if process is not None:
        if process.pid is not None and process.poll() is None:
            process.terminate()
        time.sleep(2)
        if process.pid is not None and process.poll() is None:  # If process has not terminated
            process.kill()    

def watchdog():

    try:
        aidj = "aidj.py"
        CheckTargetScript(aidj)
        process=None

        while True:
            time.sleep(5)
            if not IsProcessRunning(aidj):
                print(aidj, " has stopped. Restarting...")
                process = StartProcess(aidj)
            else:
                print(aidj, " is still running")

    except KeyboardInterrupt:

        print("Program interrupted by user.")
        StopProcess(process)

    except Exception as e:

        print(f"An error occurred: {e}")
        StopProcess(process)

if __name__ == "__main__":
    watchdog()
