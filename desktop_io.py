import subprocess as sp
import numpy as np

# http://stackoverflow.com/questions/6766333/capture-windows-screen-with-ffmpeg
# https://ffmpeg.org/ffmpeg-devices.html#gdigrab
# http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/

class ScreenCaster(object):


    def cast_screen(self, center_x = 400, center_y = 200, width = 640, height = 480):
        self.width = width
        self.height = height
        self.command = ['ffmpeg',
                   '-f', 'gdigrab',
                   '-offset_x', str(center_x),
                   '-offset_y', str(center_y),
                   '-video_size', str(width) + 'x' + str(height),
                   '-show_region', '1',
                   '-i', 'desktop',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vcodec', 'rawvideo',
                   'pipe:1']

        self.pipe = sp.Popen(self.command, stdin= sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 8)

    def end_cast(self):
        self.pipe.communicate(b'q\n')

    def terminate_cast(self):
        self.pipe.terminate()

    def kill_cast(self):
        self.pipe.kill()

    def get_image(self):
        raw_image = self.pipe.stdout.read(self.width * self.height * 3)
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape((self.width, self.height, 3))
        self.pipe.stdout.flush()
        return image

if __name__ == '__main__':
    sc = ScreenCaster()
    sc.cast_screen()
    for i in range(0,100):
        print(i,sc.get_image())

    sc.end_cast()