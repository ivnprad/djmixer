import os
import random

from FileHandling import GetSongData,SaveToJson, songsListFile

from enum import Enum

# Define the enum
class Pattern(Enum):
    PATTERN_ASCENDING = 1
    PATTERN_DESCENDING = 2
    PATTERN_ENUM_END = 3

#songsListFile = "songsList.json"

# Create a seed with true randomness from the operating system
trueRandomSeed = int.from_bytes(os.urandom(4), 'big')
random.seed(trueRandomSeed)

def GeneratePattern(songData):
    sortedSongPathBeat = dict(sorted(songData.items(), key=lambda item: item[1], reverse=False))
    
    sortedSongs = list(sortedSongPathBeat.keys())
    return sortedSongs

#TODO FirstGenerativePattern does not respect 2.5-7 BPM transition. Need debugging
def FirstGenerativePattern(songData, pattern=Pattern.PATTERN_ASCENDING):
    
    """
    Generates a playlist from a dictionary of songs with their BPM (Beats Per Minute).

    The function sorts the input songs by BPM, then creates a playlist starting from a random 
    song among the ten with the lowest BPM. It gradually adds songs to the playlist, ensuring 
    each subsequent song has a BPM slightly higher (between 2.5 to 7 BPM more) than the last. 
    The process stops when it reaches the song with the highest BPM in the input or when no 
    suitable songs are left to maintain the BPM range progression.

    Parameters:
    - songData (dict): A dictionary where keys are song names and values are their BPM.

    Returns:
    - list: A list of song names representing the generated playlist.

    Note:
    - The function assumes songData contains more than 10 songs; behavior is undefined otherwise.
    - Uses random selection, so output can vary in different calls.
    - Includes TODOs for future improvements: checking size of songData and modifying sortedSongsBeat subspan.

    Example:
    songData = {"Song A": 120, "Song B": 110, ...}
    playlist = FirstGenerativePattern(songData)
    print("Generated Playlist:", playlist)
    """
    sortedSongsBeat = dict(sorted(songData.items(), key=lambda item: item[1], reverse=False ))

    filteredSongsBeat = None
    
    songsToBePlayed = GetSongData(songsListFile)
    if bool(songsToBePlayed) == True:
        filteredSongsBeat = {key: value for key, value in sortedSongsBeat.items() if key not in songsToBePlayed}
    else:
        filteredSongsBeat = sortedSongsBeat
    

    _, limitBPM = list(filteredSongsBeat.items())[-1]


    #TODO check if dict has more than 10 times
    numberOfInitiallyItems = 10
    firstTenTimes = list (filteredSongsBeat.items()) [:numberOfInitiallyItems]
    randomItem = random.choice(firstTenTimes) 
    initialSongToBePlayed = {randomItem[0]:randomItem[1]}

    
    generalDictOfSongsToBePlayed=initialSongToBePlayed
    keepAdding = True
    while keepAdding:
        _, currentBPM = list(generalDictOfSongsToBePlayed.items())[-1]
        lowerLimitBPM = 2.5 + currentBPM
        uppperLimitBPM = 7 + currentBPM

        if currentBPM+5>limitBPM:
            break

        # TODO subspan sortedSongsBeat
        possiblyNextSongs = {key: value for key, value in filteredSongsBeat.items() if lowerLimitBPM < value < uppperLimitBPM}

        if bool(possiblyNextSongs) == False:
            break

        for songToBePlayed in generalDictOfSongsToBePlayed.keys():
            if songToBePlayed in possiblyNextSongs:
                possiblyNextSongs.pop(songToBePlayed)

        nextSong = random.choice(list(possiblyNextSongs.items()))

        generalDictOfSongsToBePlayed[nextSong[0]] = nextSong[1]

    # for key, value in generalDictOfSongsToBePlayed.items():
    #     print(f"song: {key}, bpm: {value}")

    itsSongsToBePlayed = None
    if pattern==Pattern.PATTERN_ASCENDING:
        itsSongsToBePlayed = generalDictOfSongsToBePlayed
    elif pattern==Pattern.PATTERN_DESCENDING:
        itsSongsToBePlayed = dict(reversed(generalDictOfSongsToBePlayed.items()))

    return itsSongsToBePlayed




