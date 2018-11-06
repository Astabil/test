import threading
import time
import queue
from multiprocessing import Queue
import cv2
frames = Queue(10)

class ImageGrabber(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.ID=ID
        self.cam=cv2.VideoCapture(ID)

    def Run(self):
        global frames
        while True:
            ret,frame=self.cam.read()
            frames.put(frame)
            time.sleep(0.1)


class Main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global frames
        while True:
            if(not frames.empty()):
                self.Currframe=frames.get()



grabber = ImageGrabber(0)
main = Main()

grabber.start()
main.start()
main.join()
grabber.join()