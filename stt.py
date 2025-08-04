import sounddevice as sd
import vosk
import sys
import queue
import json

class STT:
    def __init__(self, modelpath: str = "model", samplerate: int = 16000):
        self.__REC__ = vosk.KaldiRecognizer(vosk.Model(modelpath), samplerate)
        self.__Q__ = queue.Queue()
        self.__SAMPLERATE__ = samplerate

    
    def q_callback(self, indata, _, __, status):
        if status:
            print(status, file=sys.stderr)
        self.__Q__.put(bytes(indata))

    def listen(self, executor: callable):
        with sd.RawInputStream(
                samplerate=self.__SAMPLERATE__, 
                blocksize=8000, 
                device=1, 
                dtype='int16',
                channels=1, 
                callback=self.q_callback
            ):
            while True:
                data = self.__Q__.get()
                if self.__REC__.AcceptWaveform(data):
                    executor(json.loads(self.__REC__.Result())["text"])