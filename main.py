import asyncio
from AudioProcessing import PlayAsync, CalculateTransition
from FileHandling import UpdateJsonWithSong
from Utilities import GetModLength
from Core.CreateListOfSongs import GetPreviousSessionSongs, CreateNewListOfSongs
from Logging.MainLogger import mainLogger
import argparse

#TODO use GUI to display songs with name of the song and artist. Use this 
#TODO .json the ASCII coding does not accept "tilde" this is range 
#TODO use separate thread to analize songs who are new so it does not take too long to load 
#TODO bug If songsplayed.json is too big creating a new list 

async def WaitForTasksToFinish(tasks):
    await asyncio.gather(*(task for task in tasks if task))

async def aidj(resume=None):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--resume", help="An optional parameter", default=None)
    
    args = parser.parse_args()

    asyncio.run(aidj(args.resume))
