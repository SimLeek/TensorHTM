import numpy as np
import subprocess as sp
import socket as sock
import cv2
# http://stackoverflow.com/questions/6766333/capture-windows-screen-with-ffmpeg
# https://ffmpeg.org/ffmpeg-devices.html#gdigrab
#http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
class ScreenCaster(object):
    def cast_screen(self, cam_list):
        self.cam_list = cam_list
        self.start_port = 12345
        #replace ffmpeg with one stream of whole desktop/cam
        #and process with opencv
        self.command = ['ffmpeg', '-f',
                        'dshow',
                        '-i', 'video=screen-capture-recorder',
                        '-filter:v', 'scale=iw/2:-1',
                        '-crf', '0',
                        '-f', 'rtp',
                        '-sdp_file', 'stream.sdp',
                        'rtp://localhost:12345']

        # self.tf.write("do it")
        self.p = sp.Popen(self.command, stdout=sp.PIPE, stdin=sp.PIPE)
        self.cv_cam=cv2.VideoCapture()
        self.cv_cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cv_cam.open("stream.sdp")

    def end_cast(self):
        self.p.communicate(b'q\n')

    def terminate_cast(self):
        self.p.terminate()
        self.cv_cam.release()

    def get_image(self):
        #
        #Important! If other algorithms stop this from being executed often, output will be buggy
        # currently does not work with point fields. Use texture mapping.
        #Todo: consider putting this read method on its own thread.
        raw_image = self.cv_cam.read()
        raw_image = (True, cv2.cvtColor(raw_image[1], cv2.COLOR_BGR2RGB))
        images = []

        if raw_image[0] is False:
            return [(False, raw_image)]

        for i in range(len(self.cam_list)):
            height, width, channels = raw_image[1].shape
            top_left_x = int(max(self.cam_list[i]['center_x'] - self.cam_list[i]['capture_width']/2, 0))
            top_left_y = int(max(self.cam_list[i]['center_y'] - self.cam_list[i]['capture_height'] / 2, 0))

            bottom_right_x = int(min(width, top_left_x + int(self.cam_list[i]['capture_width'])))
            bottom_right_y = int(min(height, top_left_y + int(self.cam_list[i]['capture_height'])))

            cropped_image = raw_image[1][
                            top_left_y:bottom_right_y,
                            top_left_x:bottom_right_x
                            ]
            shrunken_image = cv2.resize(cropped_image, (self.cam_list[i]['output_width'],
                                                        self.cam_list[i]['output_height']),
                                        interpolation = cv2.INTER_AREA)
            # via: http://stackoverflow.com/a/32455269
            gaussian_3 = cv2.GaussianBlur(shrunken_image, (9, 9), 10.0)
            unsharp_image = cv2.addWeighted(shrunken_image, 1.5, gaussian_3, -0.5, 0)
            images.append((True, unsharp_image))

        return images

if __name__ == '__main__':
    sc = ScreenCaster()
    sc.cast_screen()
    for i in range(0,1000000000):
        sc.get_image()

    sc.end_cast()