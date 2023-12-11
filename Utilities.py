


def GetModLength(iterable):
    modLength = len(iterable)
    if (modLength % 2 == 0):
        return modLength
    else:
        return modLength-1