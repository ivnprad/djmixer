import asyncio
from Core.PlaySongs import PlaySongsAlt
import threading
import argparse
from threading import Event
import time

#TODO use GUI to display songs with name of the song and artist. Use this 
#TODO use separate thread to analize songs who are new so it does not take too long to load 
#TODO bug If songsplayed.json is too big creating a new list 

def RunAsyncInThread(resume=None,stopEvent=None):
    asyncio.run(PlaySongsAlt(resume,stopEvent))

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--resume", action="store_true",help="An optional parameter", default=None)
    args = parser.parse_args()

    stopEvent =  asyncio.Event()
    thread = threading.Thread(target=RunAsyncInThread, args=(args.resume,stopEvent))
    thread.start()
    time.sleep(20)
    stopEvent.set()
    thread.join()  

# async def playasync():
#     resume=False
#     cancelEvent = asyncio.Event()
#     task_coroutine = asyncio.create_task(PlaySongsAlt(resume,cancelEvent))
#     await asyncio.sleep(60)
#     print("System is doing other work")
#     cancelEvent.set()
#     print("Cancel event set")
#     await task_coroutine

if __name__ == "__main__":
    main()
    #asyncio.run(playasync())

