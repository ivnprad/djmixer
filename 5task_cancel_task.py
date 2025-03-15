import asyncio
from time import sleep

def Play(taskname,stopEvent):
    while True:
        print(f"{taskname} async thread playing music")
        sleep(5)
        if stopEvent.is_set():
            print("Stopping early.")
            break

async def PlayAsyncWithTrhead(taskname,stopEvent):
    await asyncio.to_thread(Play, taskname,stopEvent)

async def PlayAsync(taskName,event_for_wait, event_for_set):
    while True:
        # Wait for an event to start playing
        await event_for_wait.wait()
        event_for_wait.clear()  # Clear the event to wait for the next trigger

        print(f"{taskName} async Playing...")
        # Simulate the playing
        await asyncio.sleep(5)  # Assuming there's some work here

        # Set the event to notify the other task to start playing
        event_for_set.set()

async def leftjob():
    print("leftJob started")
    await asyncio.sleep(180) 
    print("leftJob completed")
    return "rightJob result"

async def rightjob():
    print("rightJob started")
    await asyncio.sleep(180)  
    print("rightJob completed")
    return "rightJob result"

async def task(cancel_event):

    leftDeckEvent = asyncio.Event()
    rightDeckEvent = asyncio.Event()
    leftDeckEvent.set()

    print("Task started")
    #rightDeckTaskWithThread = asyncio.create_task(PlayAsyncWithTrhead("right async with thread",cancel_event))
    #leftDeckTaskWithThread = asyncio.create_task(PlayAsyncWithTrhead("left async with thread",cancel_event))
    leftDeckTask = asyncio.create_task(PlayAsync("Left Desk Task",leftDeckEvent, rightDeckEvent))
    rightDeckTask = asyncio.create_task(PlayAsync("Right Desk Task",rightDeckEvent, leftDeckEvent))
    left_job_task = asyncio.create_task(leftjob())
    right_job_task = asyncio.create_task(rightjob())
    cancel_task = asyncio.create_task(cancel_event.wait())

    # Run job and wait for the cancel event, continue as soon as one of them is done
    done, pending = await asyncio.wait(
        [leftDeckTask,rightDeckTask,left_job_task, right_job_task,cancel_task],
        return_when=asyncio.FIRST_COMPLETED  # Proceeds when the first task completes
    )

    for future in done:
        if future.done():
            try:
                result = await future  # This retrieves the result if it's not an event wait
                print(f"Completed first with result: {result}")
            except asyncio.CancelledError:
                print("Task was cancelled.")

    for future in pending:
        future.cancel()
        print("Canceled pending task")

async def main():
    cancel_event = asyncio.Event()
    task_coroutine = asyncio.create_task(task(cancel_event))
    await asyncio.sleep(20)
    print("System is doing other work")
    cancel_event.set()
    print("Cancel event set")
    await task_coroutine

# Run the main coroutine
asyncio.run(main())
