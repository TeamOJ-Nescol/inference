import cv2
import platform

class Camera:
    def __init__(self, cam_num):
        self.cam_num = cam_num

        if platform.system() == "Darwin":
            self.cam = cv2.VideoCapture(self.cam_num, cv2.CAP_AVFOUNDATION)
        else:
            self.cam = cv2.VideoCapture(self.cam_num)

        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open camera {self.cam_num}")

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Reruns to make sure camera is not blured
        for i in range(10):
            self.cam.read()
            
        ok, self.start_frame = self.cam.read()
        if not ok or self.start_frame is None:
            self.cam.release()
            raise RuntimeError(f"Could not read initial frame from camera {self.cam_num}")

        ok, self.cur_frame = self.cam.read()
        if not ok or self.cur_frame is None:
            self.cam.release()
            raise RuntimeError(f"Could not read current frame from camera {self.cam_num}")

    def get_cur_frame(self):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ok, self.cur_frame = self.cam.read()
        if not ok or self.cur_frame is None:
            return None

        return self.cur_frame