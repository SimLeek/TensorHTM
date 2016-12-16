import numpy as np
import subprocess as sp
import tempfile as tf
# http://stackoverflow.com/questions/6766333/capture-windows-screen-with-ffmpeg
# https://ffmpeg.org/ffmpeg-devices.html#gdigrab
# http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/

class ScreenCaster(object):

    def cast_screen(self, center_x = 400, center_y = 200, height = 100, width = 100):
        self.height = height
        self.width = width
        self.center_x = center_x
        self.center_y = center_y
        #todo: set registry options for screen-capture-recorder before start
        self.command = ['ffmpeg', '-f',
                        'dshow',
                   #'-framerate', '50',
                   '-i', 'video=screen-capture-recorder',
                   '-filter:v', 'crop=' + str(width) + ':' + str(height) + ':' + str(center_x) + ':' + str(center_y)+', '+
                   'unsharp=13:13:5, unsharp=13:13:5',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vcodec', 'rawvideo',
                   'pipe:1']

        self.tfile = tf.NamedTemporaryFile(mode = 'w+b', buffering = 0, delete=False)
        # self.tf.write("do it")
        self.p = sp.Popen(self.command, stdout=self.tfile, stdin=sp.PIPE)
        self.frame_number=0

    def end_cast(self):
        self.p.communicate(b'q\n')

    def terminate_cast(self):
        self.p.terminate()
        self.tfile.close()

    def get_image(self):
        # https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
        # (Tell Anders I love them.)
        self.tfile.seek(self.frame_number * self.height * self.width * 3)
        # raw_image = self.tfile.read(self.width*self.height*3)

        raw_image = self.tfile.read(self.height * self.width * 3)

        # print(self.pipe.stderr.read())
        try:
            image = np.fromstring(raw_image, dtype='uint8')
            image = image.reshape((self.height, self.width, 3))
            self.frame_number += 1
            # print(self.frame_number)
            # self.p.stdout.flush()
            self.tfile.flush()
            return image
        except ValueError:
            return False


if __name__ == '__main__':
    sc = ScreenCaster()
    sc.cast_screen()
    for i in range(0,1000000000):
        sc.get_image()

    sc.end_cast()