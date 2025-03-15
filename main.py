import asyncio
from Core.PlaySongs import PlaySongs
import threading
import argparse
from threading import Event
import time

#TODO use GUI to display songs with name of the song and artist. Use this 
#TODO .json the ASCII coding does not accept "tilde" this is strange 
#TODO use separate thread to analize songs who are new so it does not take too long to load 
#TODO bug If songsplayed.json is too big creating a new list 

def RunAsyncInThread(resume=None,stopEvent=None):
    asyncio.run(PlaySongs(resume,stopEvent))

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--resume", action="store_true",help="An optional parameter", default=None)
    args = parser.parse_args()

    stopEvent = Event()
    thread = threading.Thread(target=RunAsyncInThread, args=(args.resume,stopEvent))
    thread.start()
    time.sleep(20)
    stopEvent.set()
    thread.join()  

if __name__ == "__main__":
    main()

