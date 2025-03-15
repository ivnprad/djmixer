import asyncio
from threading import Event
from time import sleep
from asyncio import to_thread

def Play(stopEvent):
    # Check every second if the event is set
    #for _ in range(180):
    while True:
        if stopEvent.is_set():
            print("Stopping early.")
            break
        sleep(1)
        print("doing sth next round")
    else:
        print("Completed without interruption.")

async def PlayAsync():
    stopEvent = Event()
    task = asyncio.create_task(to_thread(Play, stopEvent))
    # You can now control when to stop
    # For example, let's stop it after 5 seconds
    await asyncio.sleep(5)
    stopEvent.set()
    await task  # Wait for the thread to finish

# Run the async function
asyncio.run(PlayAsync())
