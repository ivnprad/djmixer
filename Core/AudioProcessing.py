import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide" # Set PYGAME_HIDE_SUPPORT_PROMPT to hide the support prompt
import pygame
import librosa
import tempfile
import soundfile as sf
from pydub import AudioSegment
import os
import time
import simpleaudio as sa
from asyncio import to_thread
from FileHandling import DeleteAndCreate
from Logging.MainLogger import mainLogger
# TODO Continuous music beat

silenceThresholdInDbfs= -35
milisecondsToSeconds = 1/1000 

def Play(filePath, logUntilThisLimit,stopEvent,startPos=0):
    # Load the file
    song = AudioSegment.from_file(filePath)
    secondsToMilisecondsFactor = 1000
    startPosMS = startPos * secondsToMilisecondsFactor
    playbackSegment = song[startPosMS:]

    # Convert the segment to raw audio data. Get the sample rate. Get the number of channels
    rawData = playbackSegment.raw_data
    sampleRate = playbackSegment.frame_rate
    numChannels = playbackSegment.channels
    playObj = sa.play_buffer(rawData, numChannels, 2, sampleRate)

    startTime = time.time()
    while playObj.is_playing():
        currentTime = time.time() - startTime + startPos 
        if currentTime<logUntilThisLimit:
            mainLogger.debug(f"{filePath[-30:]} Current playback position: {currentTime:.2f} seconds")
            DeleteAndCreate(filePath,currentTime)
        time.sleep(0.5)
        if stopEvent.is_set():
            playObj.stop()
            break
    
    mainLogger.debug("playing done")

    return currentTime

async def PlayAsync(filePath, logUntilThisLimit,stopEvent,startPos=0):
    await to_thread(Play, filePath, logUntilThisLimit, stopEvent,startPos)

# Calculate beats 
def CalculateBeats(mp3Path):
    audio, sr = librosa.load(mp3Path, sr=None)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
        sf.write(temp_wav.name, audio, sr)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

    return tempo

# Get mp3 from file
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

def ConvertM4AtoMp3(folderPath):
    try:
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.endswith((".m4a")):
                    mp3file = file[:-4] + ".mp3"
                    outputFile = root + "/"+ mp3file
                    if os.path.exists(outputFile):
                        continue
                    inputFile = root + "/"+ file
                    audio = AudioSegment.from_file(inputFile, format="m4a")
                    audio.export(outputFile, format="mp3")
    except Exception as e:
         raise ValueError("converting .m4a to .mp3 raise this exception ") from e
         

def GetSongWithAudioSegment(song):

    audioSegmentSong = None
    if not isinstance(song, AudioSegment):
        try:
            audioSegmentSong = AudioSegment.from_file(song)
            return audioSegmentSong
        except Exception as e:
            raise ValueError("Provided 'song' must be an AudioSegment object or a valid file path") from e
    
    return song
   
# Detect silecente portions of with threshold of -35
def DetectSilencePortionsOfSong(song):

    song = GetSongWithAudioSegment(song)
    durationMs = len(song)

    # Find start_ms
    for nonSilentStartTimeInMs in range(durationMs):
        if song[nonSilentStartTimeInMs].dBFS > silenceThresholdInDbfs:
            break

    # Find end_ms
    for nonSilentEndTimeInMs in range(durationMs - 1, -1, -1):
        if song[nonSilentEndTimeInMs].dBFS > silenceThresholdInDbfs:
            break

    songDuration = durationMs*milisecondsToSeconds
    nonSilentStartTime = nonSilentStartTimeInMs*milisecondsToSeconds
    nonSilentEndTime  = nonSilentEndTimeInMs*milisecondsToSeconds
    silenceAtEndDuration = songDuration-nonSilentEndTime
    
    return nonSilentStartTime,nonSilentEndTime,silenceAtEndDuration, songDuration

def GetNonSilentStartTime(song):
    song = GetSongWithAudioSegment(song)
    durationMs = len(song)

    for nonSilentStartTimeInMs in range(durationMs):
        if song[nonSilentStartTimeInMs].dBFS > silenceThresholdInDbfs:
            break    
    
    return nonSilentStartTimeInMs*milisecondsToSeconds

def GetNonSilentEndTime(song):
    song = GetSongWithAudioSegment(song)
    durationMs = len(song)    

    for nonSilentEndTimeInMs in range(durationMs - 1, -1, -1):
        if song[nonSilentEndTimeInMs].dBFS > silenceThresholdInDbfs:
            break
    
    return nonSilentEndTimeInMs*milisecondsToSeconds

def GetSongDuration(song):
    song = GetSongWithAudioSegment(song)
    return len(song)*milisecondsToSeconds     

def GetSilenceAtEndDuration(song):
    return  GetSongDuration(song) - GetNonSilentEndTime(song)


def CalculateTransition(currentSong,nextSong):
    currentDeckSong = AudioSegment.from_file(GetMP3FromFile(currentSong))    
    silenceAtEndDuration = GetSilenceAtEndDuration(currentDeckSong)
    currentSongDuration = GetSongDuration(currentDeckSong)

    nextDeckSong =AudioSegment.from_file(GetMP3FromFile(nextSong))
    nextStart = GetNonSilentStartTime(nextDeckSong)

    crossfade = silenceAtEndDuration+nextStart 
    delay = currentSongDuration-crossfade

    mainLogger.debug(f"currentSongDuration {currentSongDuration} seconds. Crossfade {crossfade} seconds. Delay {delay} seconds")

    return delay



