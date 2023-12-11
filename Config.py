import threading
import queue
from time import sleep
from pydub import AudioSegment
from pydub.playback import play
import os
import asyncio
import librosa
import tempfile
import soundfile as sf
import json
import tkinter as tk
from tkinter import filedialog

#TODO fade in - fade out
#TODO Time-Stretching the Songs Overlapping and Mixing the Songs

songBeatsFile = "songBeats.json"

# Calculate beats 
def CalculateBeats(mp3Path):
    audio, sr = librosa.load(mp3Path, sr=None)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
        sf.write(temp_wav.name, audio, sr)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

    return tempo

# Get a lenth multiple of 2 
def GetModLength(iterable):
    modLength = len(iterable)
    if (modLength % 2 == 0):
        return modLength
    else:
        return modLength-1

# Convert File to mp3
def GetMP3FromFile(filePath):
    root, extension = os.path.splitext(filePath)

    if extension.lower() == '.mp3':
        return filePath
    elif extension.lower() == '.m4a':
        audio = AudioSegment.from_file(filePath, format="m4a")
        outputFile = root + '.mp3'
        if os.path.exists(outputFile):
            return outputFile
        audio.export(outputFile,format="mp3")
        return outputFile
    else:
        return None

# List Files in folder recursively    
def ListFilesInFolderRecursively(folderPath):
    try:
        fileList = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.endswith((".mp3", ".m4a")):
                    fileList.append(os.path.join(root, file))
                #elif file.endswith(".m4a"):
                    #fullFilePath = os.path.join(root,file)
                    #fileList.append(GetMP3FromFile(fullFilePath))
        return fileList
    except Exception as e:
        return f"An error occurred: {e}"  

# Load json data and get dict of song file paths with betas 
def GetSongData(filename=songBeatsFile):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Update Json Data and save it 
def SaveToJson(songData, filename=songBeatsFile):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    data.update(songData)
    
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

# Detect silecente portions of with threshold of -35
def DetectSilencePortionsOfSong(song):
    #song = AudioSegment.from_file(songPath)
    # Define silence threshold (in dBFS)
    if not isinstance(song, AudioSegment):
        try:
            song = AudioSegment.from_file(song)
        except Exception as e:
            raise ValueError("Provided 'song' must be an AudioSegment object or a valid file path") from e

    silence_threshold = -35  # This is an example value, adjust based on your needs
    duration_ms = len(song)

    for start_ms in range(duration_ms):
        if song[start_ms].dBFS > silence_threshold:
            break

    for end_ms in range(duration_ms - 1, -1, -1):
        if song[end_ms].dBFS > silence_threshold:
            break

    start = start_ms/1000
    end  = end_ms/1000
    toEnd = (len(song)-end_ms)/1000
    songDuration = duration_ms/1000

    return start,end,toEnd, songDuration

def SelecDirectory():
    # Create a root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file selection dialog
    dirPath = filedialog.askdirectory()

    # Print the selected file path
    print(f"Selected directory: {dirPath}")

    # Close the root window
    root.destroy()

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

    sortedSongPathBeat = dict(sorted(songData.items(), key=lambda item: item[1], reverse=False))
    sortedSongs = list(sortedSongPathBeat.keys())

    leftDeckTask = None
    rightDeckTask = None
    Tasks = [leftDeckTask,rightDeckTask]
    leftCrossfade = 0
    rightCrossfade = 0
    iterationStep = 2

    try:
        songsList = sortedSongs
        for iter in range(0, GetModLength(songsList), iterationStep):
            if iter==len(songsList) or iter+1==len(songsList) or iter+2==len(songsList): #for the fade-in fade-out we need to get the next 3 sogns
                break

            leftSongPath = songsList[iter]
            rightSongPath = songsList[iter+1]

            leftDeckSong = AudioSegment.from_file(GetMP3FromFile(leftSongPath))
            leftStart,leftEnd,leftToEnd,leftDeckSongDuration = DetectSilencePortionsOfSong(leftDeckSong)

            rightDeckSong =AudioSegment.from_file(GetMP3FromFile(rightSongPath))
            rightStart,rightEnd,rightToEnd,rightDeckSongDuration = DetectSilencePortionsOfSong(rightDeckSong)

            print(leftSongPath)
            print(f" Left Song. Start {leftStart} End {leftEnd} toEnd {leftToEnd} duration {leftDeckSongDuration}")

            print(rightSongPath)
            print(f" Right song. Start {rightStart} End {rightEnd} toEnd {rightToEnd} duration {rightDeckSongDuration}")

            # The overlapping should occur cutting the silence of the first song and the silence of the rightDeckson
            leftCrossfade = leftToEnd+rightStart # 10 seconds default, end silence of this song + start silence of the next song 
            print(f"leftCrossfade {leftCrossfade} seconds ")
            leftDelay = leftDeckSongDuration-leftCrossfade
            leftDeckTask = asyncio.create_task(PlaySongAsync(leftDeckSong))
            if rightDeckTask: # If right deck keeps playing let it play
                await rightDeckTask
            await asyncio.sleep(leftDelay-rightCrossfade) # TODO recalculate this in cycle

            # just tentative 
            nextSongPath = songsList[iter+2]
            nextDeckSong = AudioSegment.from_file(GetMP3FromFile(nextSongPath))
            nextStart,_,_,_ = DetectSilencePortionsOfSong(nextDeckSong)

            rightCrossfade = rightToEnd + nextStart # 10 seconds default, end silence of this song + start silence of the next song
            print(f"rightCrossfade {rightCrossfade} seconds ") 
            rightDelay = rightDeckSongDuration-rightCrossfade
            rightDeckTask = asyncio.create_task(PlaySongAsync(rightDeckSong))
            if leftDeckTask:
                await leftDeckTask
            await asyncio.sleep(rightDelay-leftCrossfade)# TODO recalculate this 

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
