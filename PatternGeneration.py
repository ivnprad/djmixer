import os
import random

# Create a seed with true randomness from the operating system
trueRandomSeed = int.from_bytes(os.urandom(4), 'big')
random.seed(trueRandomSeed)


def GeneratePattern(songData):
    sortedSongPathBeat = dict(sorted(songData.items(), key=lambda item: item[1], reverse=False))
    
    sortedSongs = list(sortedSongPathBeat.keys())
    return sortedSongs

def FirstGenerativePattern(songData):
    sortedSongsBeat = dict(sorted(songData.items(), key=lambda item: item[1], reverse=False ))
    lastSong, limitBPM = list (sortedSongsBeat.items())[-1]

    print(f"lst song: {lastSong}")
    print(f"bpm: {limitBPM}")

    #TODO check if dict has more than 10 times
    numberOfInitiallyItems = 10
    firstTenTimes = list (sortedSongsBeat.items()) [:numberOfInitiallyItems]
    randomItem = random.choice(firstTenTimes) 

    dictOfSongsToBePlayed = {randomItem[0]:randomItem[1]}
    
    keepAdding = True
    while keepAdding:
        _, currentBPM = list(dictOfSongsToBePlayed.items())[-1]
        lowerLimitBPM = 2.5 + currentBPM
        uppperLimitBPM = 7 + currentBPM

        if currentBPM+5>limitBPM:
            break

        # TODO subspan sortedSongsBeat
        possiblyNextSongs = {key: value for key, value in sortedSongsBeat.items() if lowerLimitBPM < value < uppperLimitBPM}

        if bool(possiblyNextSongs) == False:
            break

        for songToBePlayed in dictOfSongsToBePlayed.keys():
            if songToBePlayed in possiblyNextSongs:
                possiblyNextSongs.pop(songToBePlayed)

        nextSong = random.choice(list(possiblyNextSongs.items()))

        dictOfSongsToBePlayed[nextSong[0]] = nextSong[1]

    return list(dictOfSongsToBePlayed.keys())


# ascending order False , descending order True

# 1.- sort in ascending order
# 2.- pick randonmly one song out of first half (maybe 10 ) of the list 
# 3.- save list of already played songs to "listOfSongsToBePlayed" 
# 4.- create "possiblyNextSongList" 
# look for all songs which are song item BPM has between 2.5 BPM - 5 BPM more tha last 
# song 
# 5.- if "possiblyNextSongList" is empty. End of list achieve return "listOfSongsToBePlayed to be played"
# 6.- if "possiyblyNextSongList" is not empty. 
# crate Substract "possiblyNextSongList"-="listOfSongsToBePlayed"
# 7.- pick Randomly one song of "possiblyNextSongList" and add it to "ListOfSongsToBePlayed"
# 9 .- 

