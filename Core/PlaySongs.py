import asyncio

from Core.AudioProcessing import PlayAsync, CalculateTransition
from FileHandling import UpdateJsonWithSong
from Utilities import GetModLength
from Core.CreateListOfSongs import GetPreviousSessionSongs, CreateNewListOfSongs
from Logging.MainLogger import mainLogger
from threading import Event

async def WaitForTasksToFinish(tasks):
    await asyncio.gather(*(task for task in tasks if task))

async def PlaySongs(resume=None,stopEvent=None):
    mainLogger.info(" starting ")
    stopEvent = Event()
    leftDeckTask = None
    rightDeckTask = None
    Tasks = [leftDeckTask,rightDeckTask]
    leftCrossfade = 0
    rightCrossfade = 0
    iterationStep = 2

    try:
        songsList = None
        resumePosition = 0

        if resume:
            subsetList,resumePosition = GetPreviousSessionSongs()
            songsList = subsetList
        else: 
            songsList = CreateNewListOfSongs()

        for iter in range(0, GetModLength(songsList), iterationStep):
            if stopEvent.is_set():
                print("Stopping please...")
                break
            leftDeckIdx=iter
            rightDeckIdx=iter+1
            rightDeckOverlapIdx=iter+2
            if rightDeckOverlapIdx==len(songsList): 
                break

            leftSongPath = songsList[leftDeckIdx]
            rightSongPath = songsList[rightDeckIdx]

            mainLogger.info("playing: " + leftSongPath.split('/')[-1])
            UpdateJsonWithSong(leftSongPath)
            leftDelay = CalculateTransition(leftSongPath,rightSongPath)
            leftDeckTask = asyncio.create_task(PlayAsync(leftSongPath,leftDelay,stopEvent,resumePosition))

            await asyncio.sleep(5)
            stopEvent.set()
  
            if rightDeckTask: 
                await rightDeckTask
            if resumePosition != 0:
                leftDelay-=resumePosition
                resumePosition = 0

            #await asyncio.sleep(leftDelay-rightCrossfade) 
            await asyncio.sleep(5)
            # sleep_task = asyncio.create_task(asyncio.sleep(leftDelay-rightCrossfade))
            # cancel_task = asyncio.create_task(stopEvent.wait())
            # done, pending = await asyncio.wait([sleep_task, cancel_task],return_when=asyncio.FIRST_COMPLETED)
            # if sleep_task in done:
            #     print("Sleep task completed")
            # if cancel_task in done:
            #     print("Cancel task triggered")
            # for task in pending:
            #     task.cancel()
            #     try:
            #         await task
            #     except asyncio.CancelledError:
            #         print("Cancelled pending task")
    

            mainLogger.info("playing: " + rightSongPath.split('/')[-1])
            UpdateJsonWithSong(rightSongPath)
            rightDelay = CalculateTransition(rightSongPath,songsList[rightDeckOverlapIdx])
            rightDeckTask = asyncio.create_task(PlayAsync(rightSongPath,rightDelay,stopEvent))

            if leftDeckTask:
                await leftDeckTask
            #await asyncio.sleep(rightDelay-leftCrossfade)
            await asyncio.sleep(5)
            # sleep_task = asyncio.create_task(asyncio.sleep(rightDelay-leftCrossfade))
            # cancel_task = asyncio.create_task(stopEvent.wait())
            # done, pending = await asyncio.wait([sleep_task, cancel_task],return_when=asyncio.FIRST_COMPLETED)
            # if sleep_task in done:
            #     print("Sleep task completed")
            # if cancel_task in done:
            #     print("Cancel task triggered")
            # for task in pending:
            #     task.cancel()
            #     try:
            #         await task
            #     except asyncio.CancelledError:
            #         print("Cancelled pending task")
        
        await WaitForTasksToFinish(Tasks)

    except Exception as e:

        await WaitForTasksToFinish(Tasks)
        print("Caught an exception:", str(e))