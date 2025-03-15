import asyncio

async def PlayAsync(taskName,event_for_wait, event_for_set):
    while True:
        # Wait for an event to start playing
        await event_for_wait.wait()
        event_for_wait.clear()  # Clear the event to wait for the next trigger

        print(f"{taskName} Playing...")
        # Simulate the playing
        await asyncio.sleep(5)  # Assuming there's some work here

        # Set the event to notify the other task to start playing
        event_for_set.set()

async def main():
    # Create event objects
    leftDeckEvent = asyncio.Event()
    rightDeckEvent = asyncio.Event()

    # Initially trigger left deck to play first
    leftDeckEvent.set()

    # Create tasks
    leftDeckTask = asyncio.create_task(PlayAsync("Left Desk Task",leftDeckEvent, rightDeckEvent))
    rightDeckTask = asyncio.create_task(PlayAsync("Right Desk Task",rightDeckEvent, leftDeckEvent))

    # Wait for the tasks to complete (they won't in this perpetual example)
    await asyncio.gather(leftDeckTask, rightDeckTask)
    #await asyncio.gather(leftDeckTask)
    #await leftDeckEvent

# Run the main function
asyncio.run(main())
