import asyncio

from Core.AudioProcessing import PlayAsync, CalculateTransition
from FileHandling import UpdateJsonWithSong
from Utilities import GetModLength
from Core.CreateListOfSongs import GetPreviousSessionSongs, CreateNewListOfSongs
from Logging.MainLogger import mainLogger

async def WaitForTasksToFinish(tasks):
    await asyncio.gather(*(task for task in tasks if task))

async def PlaySongs(resume=None):
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
            leftDeckTask = asyncio.create_task(PlayAsync(leftSongPath,leftDelay, resumePosition))

            if rightDeckTask: 
                await rightDeckTask
            if resumePosition != 0:
                leftDelay-=resumePosition
                resumePosition = 0
            await asyncio.sleep(leftDelay-rightCrossfade) 

            mainLogger.info("playing: " + rightSongPath.split('/')[-1])
            UpdateJsonWithSong(rightSongPath)
            rightDelay = CalculateTransition(rightSongPath,songsList[rightDeckOverlapIdx])
            rightDeckTask = asyncio.create_task(PlayAsync(rightSongPath,rightDelay))

            if leftDeckTask:
                await leftDeckTask
            await asyncio.sleep(rightDelay-leftCrossfade)
        
        await WaitForTasksToFinish(Tasks)

    except Exception as e:

        await WaitForTasksToFinish(Tasks)
        print("Caught an exception:", str(e))