import asyncio
from Core.AudioProcessing import PlayAsync, CalculateTransition
from FileHandling import UpdateJsonWithSong
from Core.CreateListOfSongs import GetPreviousSessionSongs, CreateNewListOfSongs
from Logging.MainLogger import mainLogger

async def PlayAsyncAlt(songList,waitToStart, fireNextSong,stopEvent,initialIdx,resumePosition=0):
    nextSongIdx=(initialIdx+1) 
    songListCount = len(songList)
    while nextSongIdx < songListCount:
        await waitToStart.wait()
        waitToStart.clear()  

        songPath = songList[initialIdx]
        nextSongPath = songList[nextSongIdx]
        songName = songPath.split('/')[-1]

        delay = CalculateTransition(songPath,nextSongPath)
        if initialIdx==0 and resumePosition!=0:
            delay-=resumePosition

        mainLogger.info(f"Playing {songName}")
        UpdateJsonWithSong(songPath)
        deckTask = asyncio.create_task(PlayAsync(songPath,delay,stopEvent,resumePosition))
        await asyncio.sleep(delay)  

        fireNextSong.set()
        initialIdx+=2
        nextSongIdx=(initialIdx+1)
        try:
            result = await deckTask
            if not result:
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            deckTask.cancel()

async def PlaySongsAlt(resume,cancel_event):
    if not isinstance(cancel_event, asyncio.Event):
        mainLogger.error("Only asyncio.Event allowed")
        return

    resumePosition = 0

    if resume:
        subsetList,resumePosition = GetPreviousSessionSongs()
        songsList = subsetList
    else: 
        songsList = CreateNewListOfSongs()

    leftDeckEvent = asyncio.Event()
    rightDeckEvent = asyncio.Event()
    leftDeckEvent.set()

    leftDeckInitialIdx=0
    rightDeckInitialIdx=1

    mainLogger.info(f"Starting...")
    leftDeckTask = asyncio.create_task(PlayAsyncAlt(songsList,leftDeckEvent,\
                                        rightDeckEvent,cancel_event,leftDeckInitialIdx,resumePosition))
    rightDeckTask = asyncio.create_task(PlayAsyncAlt(songsList,rightDeckEvent,\
                                        leftDeckEvent,cancel_event,rightDeckInitialIdx))
    cancel_task = asyncio.create_task(cancel_event.wait())

    done, pending = await asyncio.wait(
        [leftDeckTask,rightDeckTask,cancel_task],
        return_when=asyncio.FIRST_COMPLETED  
    )

    for future in done:
        if future.done():
            try:
                result = await future  # This retrieves the result if it's not an event wait
                if not result:
                    raise ValueError("Task was not completed")
            except asyncio.CancelledError:
                print("Task was cancelled.")

    for future in pending:
        future.cancel()
        print("Canceled pending task")