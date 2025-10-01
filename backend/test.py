import dlib
from fer import FER

print("Dlib loaded:", dlib.__version__)
detector = FER()
print("FER detector ready:", detector)
