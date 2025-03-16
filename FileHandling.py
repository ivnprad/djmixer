import os
import json

from Utilities import SubSetFromKey

scriptDir = os.path.dirname(__file__)
songBeatsFile = os.path.join(scriptDir, 'ConfigurationFiles', 'songBeats.json')
songsPlayedFile = os.path.join(scriptDir, 'ConfigurationFiles', 'songsPlayed.json')
directoryFile = os.path.join(scriptDir, 'ConfigurationFiles', 'directory.json')
currentSongFile = os.path.join(scriptDir, 'ConfigurationFiles', 'currentSong.json')
songsListFile = os.path.join(scriptDir, 'ConfigurationFiles', 'songsList.json')

# List Files in folder recursively    
def ListFilesInFolderRecursively(folderPath):
    try:
        fileList = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.endswith((".mp3")):
                    fileList.append(os.path.join(root, file))
                #TODO
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
def SaveToJson(songData, filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    try:
        data.update(songData)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    
    with open(filename, "w") as file:
        json.dump(data, file, indent=4,ensure_ascii=False)

# Delete and Create json file with current song and position
def DeleteAndCreate(songPath, position, filename=currentSongFile):
    # Create the songData dictionary with songPath as key and position as value
    songData = {songPath: position}

    try:
        with open(filename, "w") as file:
            json.dump(songData, file, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# Update Json Data #TODO compare to SaveToJson and possibly use only one function
def UpdateJsonWithSong(newSong, fileName=songsPlayedFile):
    try:
        with open(fileName,"r") as file:
            songs = json.load(file)
    except FileNotFoundError:
        songs = []

    songs.append(newSong)

    with open(fileName,'w') as file:
        json.dump(songs,file,indent=4)

def ListOfSongsPlayed(fileName=songsPlayedFile):
    try:
        with open(fileName,"r") as file:
            songs = json.load(file)
    except FileNotFoundError:
        songs = []

    return songs

def GetDirectory(fileName=directoryFile):
    try:
        with open(fileName,"r") as file:
            directory=json.load(file)
    except FileNotFoundError:
        directory=None
    
    return directory

def SaveDirectory(directoryName, fileName=directoryFile):
    with open(fileName,'w') as file:
        json.dump(directoryName,file)

def DeleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

#TODO move to Core 
def GetCurrentSongAndPosition(jsonFile=currentSongFile):
    try:
        with open(jsonFile, 'r') as file:
            data = json.load(file)

        if isinstance(data, dict) and len(data) == 1:
            key = next(iter(data))  
            value = data[key]       
            return key, value
        else:
            raise ValueError("JSON does not contain exactly one key-value pair")
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON file")
    except FileNotFoundError:
        raise ValueError("File not found")

