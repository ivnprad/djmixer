import os
import json

songBeatsFile = "songBeats.json"

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