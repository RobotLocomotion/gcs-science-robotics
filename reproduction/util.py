import os

def GcsDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def FindModelFile(filename):
    assert filename[:6] == "models"
    return os.path.join(GcsDir(), filename)
