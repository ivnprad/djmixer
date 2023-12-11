import librosa
import tempfile
import soundfile as sf
from pydub import AudioSegment
import os

# Continuous music mix


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
    
# Detect silecente portions of with threshold of -35
def DetectSilencePortionsOfSong(song):
    #song = AudioSegment.from_file(songPath)

    # Define silence threshold (in dBFS)
    if not isinstance(song, AudioSegment):
        try:
            song = AudioSegment.from_file(song)
        except Exception as e:
            raise ValueError("Provided 'song' must be an AudioSegment object or a valid file path") from e

    silenceThreshold= -35  # This is an example value, adjust based on your needs
    duration_ms = len(song)

    # Find start_ms
    for start_ms in range(duration_ms):
        if song[start_ms].dBFS > silenceThreshold:
            break

    # Find end_ms
    for end_ms in range(duration_ms - 1, -1, -1):
        if song[end_ms].dBFS > silenceThreshold:
            break

    start = start_ms/1000
    end  = end_ms/1000
    toEnd = (len(song)-end_ms)/1000
    songDuration = duration_ms/1000

    return start,end,toEnd, songDuration