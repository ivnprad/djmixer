import asyncio

from Core.AudioProcessing import PlayAsync, CalculateTransition
from FileHandling import UpdateJsonWithSong
from Utilities import GetModLength
from Core.CreateListOfSongs import GetPreviousSessionSongs, CreateNewListOfSongs
from Logging.MainLogger import mainLogger
from threading import Event
import time

async def AwaitTimeoutOrEvent(stopEvent,timeout):
    startTime = time.perf_counter()
    endTime = startTime
    while (endTime - startTime) < timeout:
        if stopEvent.is_set():
            break
        await asyncio.sleep(0.1)
        endTime = time.perf_counter()

async def WaitForTasksToFinish(tasks):
    await asyncio.gather(*(task for task in tasks if task))

async def PlaySongs(resume=None,stopEvent=None):
    mainLogger.info(" starting ")
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
  
            if rightDeckTask: 
                await rightDeckTask
            if resumePosition != 0:
                leftDelay-=resumePosition
                resumePosition = 0

            await AwaitTimeoutOrEvent(stopEvent,leftDelay-rightCrossfade)

            if stopEvent.is_set():
                break
            mainLogger.info("playing: " + rightSongPath.split('/')[-1])
            UpdateJsonWithSong(rightSongPath)
            rightDelay = CalculateTransition(rightSongPath,songsList[rightDeckOverlapIdx])
            rightDeckTask = asyncio.create_task(PlayAsync(rightSongPath,rightDelay,stopEvent))

            if leftDeckTask:
                await leftDeckTask

            await AwaitTimeoutOrEvent(stopEvent,rightDelay-leftCrossfade)

        await WaitForTasksToFinish(Tasks)

    except Exception as e:

        await WaitForTasksToFinish(Tasks)
        print("Caught an exception:", str(e))