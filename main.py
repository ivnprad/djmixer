from pydub import AudioSegment
from pydub.playback import play
import asyncio
from AudioProcessing import CalculateBeats, GetMP3FromFile, DetectSilencePortionsOfSong
from FileHandling import ListFilesInFolderRecursively, GetSongData, SaveToJson
from UI import SelecDirectory
from Utilities import GetModLength
from AIExperimental.PatternGeneration import GeneratePattern

#TODO fade in - fade out
#TODO Time-Stretching the Songs Overlapping and Mixing the Songs
#TODO selects songs spotify has a lot "listens" 
#TODO select different patterns according to beats bpm

#songBeatsFile = "songBeats.json"

""" This code appears to be part of a script designed for mixing or transitioning between songs, likely for DJing or creating a continuous music mix. Here's a breakdown of its logic in simple terms:

1. **Initialization**: 
The script starts by sorting the songs into a list (`sortedSongs`).

2. **Iterating Through Songs**: 
The script iterates through the list of songs. 
The step size of each iteration (`iterationStep`) is not defined in the snippet, but it suggests the script processes multiple songs in each iteration.

3. **Breaking Condition**: If the current iteration reaches the end of the song list (or near the end), the loop stops to avoid errors.

4. **Processing Two Consecutive Songs**: 
For each iteration, the script processes two consecutive songs (`leftSongPath` and `rightSongPath`):
   - It loads each song and analyzes them to detect silent portions at the beginning and end. This information is used to determine where to best overlap or crossfade the songs.
   - The script prints out details about each song, such as the start and end of silence and the total duration.

5. **Crossfading Logic**:
   - The script calculates when to start crossfading the left song with the right song, based on the end silence of the left song and the start silence of the right song.
   - It then schedules the left song to play asynchronously, waits for the right amount of time, and then starts the crossfade.

6. **Preparing the Next Song**: 
The script also prepares the next song in the list, presumably for a smooth transition after the current pair of songs.

7. **Task Management**: 
The script manages various asynchronous tasks (like playing songs) and ensures they are properly awaited or executed.

8. **Exception Handling**: 
If any exceptions occur during execution, the script attempts to safely handle them by awaiting any pending tasks and printing the exception details.

Overall, the script seems designed to automate the process of mixing songs together, ensuring smooth transitions and crossfades based on the silent portions of the songs. 
The use of asynchronous programming suggests it's designed to handle these tasks in a non-blocking manner, which is crucial for real-time audio processing.
 """

async def PlaySongAsync(audio):
    await asyncio.to_thread(play, audio)

async def main():

    folderPath = SelecDirectory()
    songPaths = ListFilesInFolderRecursively(folderPath)
    songData = GetSongData()

    # For song list in the give directory tha end with .mp3 and are not in songData calculate Beats
    for song in songPaths:
        if not song.endswith(".mp3"):
            continue 
        if song not in songData:
            beats = round(CalculateBeats(song))
            songData[song] = beats

    SaveToJson(songData) # update list to json

    # sortedSongPathBeat = dict(sorted(songData.items(), key=lambda item: item[1], reverse=False))
    # sortedSongs = list(sortedSongPathBeat.keys())

    sortedSongs = GeneratePattern(songData)

    leftDeckTask = None
    rightDeckTask = None
    Tasks = [leftDeckTask,rightDeckTask]
    leftCrossfade = 0
    rightCrossfade = 0
    iterationStep = 2

    try:
        songsList = sortedSongs
        for iter in range(0, GetModLength(songsList), iterationStep):
            K_0=iter
            K_1=iter+1
            K_2=iter+2

            if K_2==len(songsList): #for the fade-in fade-out we need to get the next 3 sogns
                break

            leftSongPath = songsList[K_0]
            rightSongPath = songsList[K_1]

            # Loading songs
            leftDeckSong = AudioSegment.from_file(GetMP3FromFile(leftSongPath))
            rightDeckSong =AudioSegment.from_file(GetMP3FromFile(rightSongPath))

            # Analyzing songs for overlapping
            _,_,leftToEnd,leftDeckSongDuration = DetectSilencePortionsOfSong(leftDeckSong)
            rightStart,_,rightToEnd,rightDeckSongDuration = DetectSilencePortionsOfSong(rightDeckSong)

            # Calculate overlapping for the K_0 Song 
            leftCrossfade = leftToEnd+rightStart #  end silence of this song + start silence of the next song 
            leftDelay = leftDeckSongDuration-leftCrossfade
            leftDeckTask = asyncio.create_task(PlaySongAsync(leftDeckSong))

            # Wait for previous song to finish, which should have "rightCrossfade" seconds left.
            if rightDeckTask: 
                await rightDeckTask
            await asyncio.sleep(leftDelay-rightCrossfade) 

            # Calculate overlapping for the K_1 Song 
            songK_2 = AudioSegment.from_file(GetMP3FromFile(songsList[K_2]))
            startK_2,_,_,_ = DetectSilencePortionsOfSong(songK_2)
            rightCrossfade = rightToEnd + startK_2 
            rightDelay = rightDeckSongDuration-rightCrossfade
            rightDeckTask = asyncio.create_task(PlaySongAsync(rightDeckSong))

            # Wait for K_0 song to finish, which should have "leftCrossfade" seconds left.
            if leftDeckTask:
                await leftDeckTask
            await asyncio.sleep(rightDelay-leftCrossfade)


        for task in Tasks:
            if task is not None:
                await task

    except Exception as e:

        for task in Tasks:
            if task is not None:
                await task

        print("Caught an exception:", str(e))

if __name__ == "__main__":
    asyncio.run(main())
