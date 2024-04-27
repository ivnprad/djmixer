def GetModLength(iterable):
    modLength = len(iterable)
    if (modLength % 2 == 0):
        return modLength
    else:
        return modLength-1
    
def SubSetFromKey(originalDict, startKey):
    start = False
    subsetDict = {}

    for key in originalDict:
        if key == startKey or start:
            start = True
            subsetDict[key] = originalDict[key]

    return subsetDict