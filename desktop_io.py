import numpy as np
import subprocess as sp
import socket as sock
# http://stackoverflow.com/questions/6766333/capture-windows-screen-with-ffmpeg
# https://ffmpeg.org/ffmpeg-devices.html#gdigrab
#http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
class ScreenCaster(object):
    def cast_screen(self, cam_list):
        self.cam_list = cam_list
        self.start_port = 12345
        #todo: use ffmpeg's multiple output capability
        self.command = ['ffmpeg', '-f',
                        'dshow',
                        '-i', 'video=screen-capture-recorder',
                        # '-vcodec', 'libx264',
                        '-filter_complex']

        splitter = '[0:v]split='+str(len(cam_list))


        for i in range(len(cam_list)):
            splitter=splitter+'[in'+str(i+1)+']'
        splitter+=';'

        for i in range(len(cam_list)):
            splitter=splitter+'[in'+str(i+1)+']'\
                     'crop=' +\
                                str(cam_list[i]['capture_width']) +\
                                ':' +\
                                str(cam_list[i]['capture_height']) +\
                                ':' +\
                                str(cam_list[i]['center_x']-cam_list[i]['capture_width']/2) +\
                                ':' +\
                                str(cam_list[i]['center_y']-cam_list[i]['capture_height']/2) +\
                                '[midl'+str(i+1)+']'+';'+'[midl'+str(i+1)+']'+ \
                      'scale=' +\
            str(cam_list[i]['output_width']) +\
            ':' +\
            str(cam_list[i]['output_height']) +\
                      '[midr'+str(i+1)+']'+';'+'[midr'+str(i+1)+']'+ \
                      'unsharp=13:13:5'+\
                '[out'+str(i+1)+']'
            if i+1<len(cam_list):
                splitter+=';'

        self.command.append(splitter)

        for i in range(len(cam_list)):
            self.command.extend([
                   '-map', '[out'+str(i+1)+']',
                   # '-preset', 'ultrafast',
                   '-f', 'rtp',
                   # '-pix_fmt', 'rgb24',
                   '-sdp_file', 'stream.sdp'
                   'rtp://localhost:'+str(self.start_port+i)])

        str_command = ""
        for i in range(len(self.command)):
            str_command += '"'+self.command[i]+'" '
        print(str_command)

        # self.tf.write("do it")
        self.p = sp.Popen(self.command, stdout=sp.PIPE, stdin=sp.PIPE)
        self.socks=[]
        for i in range(len(cam_list)):
            self.socks.append(sock.socket(sock.AF_INET, sock.SOCK_DGRAM))
            self.socks[i].setblocking(False)
            self.socks[i].bind(('localhost', self.start_port+i))

    def end_cast(self):
        self.p.communicate(b'q\n')

    def terminate_cast(self):
        self.p.terminate()
        for i in range(len(self.socks)):
            self.socks[i].close()

    def get_image(self):
        # https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
        # (Tell Anders I love them.)
        raw_images = []
        for i in range(len(self.socks)):
            try:
                raw_images.append(self.socks[i].recvfrom(self.cam_list[i]['output_height'] * self.cam_list[i]['output_width'] * 3))
            except BlockingIOError:
                raw_images.append(b'')
        # raw_image = self.tfile.read(self.width*self.height*3)

        # print(self.pipe.stderr.read())
        images = []
        for i in range(len(self.socks)):
            try:
                images.append(np.fromstring(raw_images[i], dtype='uint8'))
                images[i] = images[i].reshape((self.cam_list[i]['output_height'], self.cam_list[i]['output_width'], 3))

                # print(self.frame_number)
                # self.p.stdout.flush()
            except ValueError:
                images.append(False)
            except AttributeError:
                images.append(False)

        return images

if __name__ == '__main__':
    sc = ScreenCaster()
    sc.cast_screen()
    for i in range(0,1000000000):
        sc.get_image()

    sc.end_cast()