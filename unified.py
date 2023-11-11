import pydub
from pydub.playback import play
import librosa
import soundfile as sf
import os

class InvalidAudioFileException(Exception):
    pass

def GetAudioFileExtension(filePath):
    _, fileExtension = os.path.splitext(filePath)
    # Normalize the extension to lowercase and remove the leading dot
    fileExtension = fileExtension.lower().lstrip('.')
    if fileExtension in ['mp3', 'm4a']:
        return fileExtension
    else:
        raise InvalidAudioFileException(f"Invalid file extension: {fileExtension}. Only 'mp3' and 'm4a' are accepted.")
    
def ConvertM4AtoMP3(filePath):
    if filePath.endswith('.m4a'):
        audio = pydub.AudioSegment.from_file(filePath, format="m4a")
        outputFile = os.path.splitext(filePath)[0] + '.mp3'
        if os.path.exists(outputFile):
            return outputFile
        audio.export(outputFile, format="mp3")
        return outputFile
    else:
        print("Returning file without converting because it is not a .m4a extension")
        return filePath

def SongBeat(songPath):
    #TODO first convert .m4a to mp3 and then to .wav? this is too expensive change later
    mp3Path = ConvertM4AtoMP3(songPath)
    audio, sr = librosa.load(mp3Path, sr=None)
    wavPath = '/Users/ivanherrera/Movies/bachata_tutorials/temp_audio.wav'
    sf.write(wavPath, audio, sr)
    audio, sr = librosa.load(wavPath, sr=None)
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    print("Tempo (BPM):", tempo)

def CrossfadeSongs(lhsSong, rhsSong, fadeDurationSec=10):
    songLHS = pydub.AudioSegment.from_file(lhsSong,GetAudioFileExtension(lhsSong))
    songRHS = pydub.AudioSegment.from_file(rhsSong,GetAudioFileExtension(rhsSong))
    fadeDurationMS = fadeDurationSec * 1000
    fadedSongLHS = songLHS.fade_out(fadeDurationSec * fadeDurationMS)  # pydub works in milliseconds
    fadedSongRHS = songRHS.fade_in(fadeDurationSec * fadeDurationMS)
    firstPart=fadedSongLHS[:-fadeDurationMS]
    secondPart=fadedSongRHS.overlay(fadedSongLHS[-fadeDurationMS:], position=0)
    final_song = firstPart + secondPart
    play(final_song)

if __name__ == "__main__":
    
    audio1path = "/Users/ivanherrera/Music/Bachata/moderna_sensual/105MBP/01 La Curiosidad (Bachata Version).m4a"
    audio2path = "/Users/ivanherrera/Movies/toworkon/songs/Antonio José - La Noche Perfecta.mp3"
    SongBeat(audio1path)
    SongBeat(audio2path)
    CrossfadeSongs(audio1path,audio2path)


