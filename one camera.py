import cv2
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print ("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False


    image_lock = threading.Lock()

    while True:

        rval, frame = cam.read()
        with image_lock:
            cv2.imshow(previewName, frame)
            cv2.waitKey(25)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("Camera 1", 1)
thread2 = camThread("Camera 2", 2)
thread1.start()
thread2.start()