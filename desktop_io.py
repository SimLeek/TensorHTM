import numpy as np
import subprocess as sp
import cv2
import tempfile as tf
import os
import warnings
import json

class OculomotorSystem(object):

    # static, because it's very unlikely this will change...
    # todo: check for changes in screen resolution, as this is an edge case that could break things.
    stream_resolutions = {}

    def __init__(self):
        self.cam_list = []
        self.start_port = 12345
        self.blink_area_multiplier = 1
        self.temporary_file = None
        self.p = None
        self.frame_number = 0
        self.cv_cam = cv2.VideoCapture()

    @classmethod
    def _get_stream_resolution(cls, cam):
        if cam['input_source'] == 'desktop-rtp-stream' or 'desktop-pipe-stream':
            if 'desktop-stream' not in cls.stream_resolutions:
                command = ['ffprobe',
                           '-v', 'quiet',
                           '-print_format', 'json',
                           '-show_streams',
                           '-f', 'dshow',
                           '-i', 'video=screen-capture-recorder']
                process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
                out, err = process.communicate()

                # Remove dshow output statement here, since it appears randomly and can't be removed via command.
                out.replace(b"leaving aero on", b"")

                screen_capture_recorder_json = json.loads(out)
                print(screen_capture_recorder_json)

                stream_resolutions = []
                for stream in screen_capture_recorder_json['streams']:
                    stream_resolutions.append((stream["width"],stream["height"]))
                cls.stream_resolutions['desktop-stream'] = stream_resolutions
                return stream_resolutions
            else:
                return cls.stream_resolutions['desktop-stream']

    @classmethod
    def _populate_cam_list(cls, cam_list):

        sources = []

        # todo: filter by inputs and then largest to smallest
        for i, cam in enumerate(cam_list):
            print('cam:', cam)

            if 'input_source' not in cam:
                if 'capture_width' in cam and cam['capture_width'] <= 800 and \
                                'capture_height' in cam and cam['capture_height'] <= 600:
                    cam['input_source'] = 'desktop-pipe-stream'
                    sources.append(('desktop-pipe-stream', i))
                else:
                    cam['input_source'] = 'desktop-rtp-stream'
                    sources.append(('desktop-rtp-stream', i))

                    # Get Stream Info:

                resolutions = cls._get_stream_resolution(cam)
                print(resolutions)
            else:
                raise NotImplementedError("Your input, '" + str(os.name) + "' isn't in the list of supported inputs.")

            if 'stream_number' not in cam:
                cam['stream_number'] = 0
            if 'capture_height' not in cam or cam['capture_height'] > resolutions[cam['stream_number']][1]:
                cam['capture_height'] = resolutions[cam['stream_number']][1]
            if 'capture_width' not in cam or cam['capture_width'] > resolutions[cam['stream_number']][1]:
                cam['capture_width'] = resolutions[cam['stream_number']][0]
            if 'output_height' not in cam:
                cam['output_height'] = cam['capture_height']
            if 'output_width' not in cam:
                cam['output_width'] = cam['capture_width']
            if 'center_x' not in cam:
                cam['center_x'] = int(cam['capture_width'] / 2)
            if 'center_y' not in cam:
                cam['center_y'] = int(cam['capture_height'] / 2)

            print('cam:', cam)

            return cam_list, sources

    @classmethod
    def _cam_to_ffmpeg(cls, cam_list, sources, blink_area_multiplier=1, compress=False):

        if os.name == 'nt':
            command = ['ffmpeg', '-f', 'dshow']
        else:
            raise NotImplementedError("Your system, '" + str(os.name) + "' isn't officially supported yet. ")
        input_source_names = []

        if any('desktop-rtp-stream' or 'desktop-pipe-stream' in sublist for sublist in sources):
            if os.name == 'nt':
                command.extend(['-i', 'video=screen-capture-recorder'])
            else:
                raise NotImplementedError("Your system, '" + str(os.name) + "' isn't officially supported yet. ")

            input_source_names.append(str(len(input_source_names)) + ':v')

        command.append('-filter_complex')

        all_pass_throughs = True
        complex_filter=''
        for input_source in input_source_names:
            complex_filter = '['+input_source+']'+'split='+str(len(sources))
            for i in range(len(sources)):
                #todo: put desktop-rtp-stream in conditional pass-through list
                if not compress and sources[i][0]=='desktop-rtp-stream':
                    complex_filter += '[out' + str(i) + ']'
                else:
                    complex_filter+='[in'+str(i)+']'
                    all_pass_throughs = False
            if not all_pass_throughs:
                complex_filter+=';'
        for i in range(len(sources)):
            # todo: replace with pass-through list check when this gets large
            if not compress and sources[i][0] == 'desktop-rtp-stream':
                continue
            # todo: implement a lambda function dictionary when this gets large
            elif sources[i][0] == 'desktop-rtp-stream':
                complex_filter +='[in'+str(i)+']'+'scale=iw/2:-1[out'+str(i)+'];'
            elif sources[i][0] == 'desktop-pipe-stream':
                first_instance = sources[i][1]

                res = cls._get_stream_resolution(cam_list[sources[i][1]])

                x_res = res[0][0]
                y_res = res[0][1]

                center_x = x_res*cam_list[first_instance]['center_x']
                center_y = y_res * cam_list[first_instance]['center_y']

                top_left_x = int(min(max(center_x - cam_list[first_instance]['capture_width'] / 2, 0),x_res))
                top_left_y = int(min(max(center_y - cam_list[first_instance]['capture_height'] / 2, 0),y_res))

                complex_filter += '[in' + str(i) + ']' + 'crop= ' +\
                    str(int(cam_list[first_instance]['capture_width']*blink_area_multiplier))+\
                    ':'+\
                    str(int(cam_list[first_instance]['capture_height']*blink_area_multiplier))+\
                    ':'+\
                    str(top_left_x)+\
                    ':'+\
                    str(top_left_y)+\
                    '[out'+str(i)+']'
            if i<len(sources)-1:
                complex_filter+=';'
        command.append(complex_filter)

        command.extend(['-crf', '0'])

        for i in range(len(sources)):
            command.extend(['-map', '[out' + str(i) + ']'])
            if sources[i][0]=='desktop-rtp-stream':
                command.extend(['-f', 'rtp',
                                '-sdp_file', 'stream.sdp',
                                'rtp://localhost:12345'])
            elif sources[i][0] == 'desktop-pipe-stream':
                command.extend(['-f', 'image2pipe',
                                '-pix_fmt', 'rgb24',
                                '-vcodec', 'rawvideo',
                                'pipe:1'])

            input_source_names = []

        return command

    def start_cast(self, cam_list):
        """Starts the streams that will supply video input to our program. Currently only supports Desktop viewing
            on Windows.

        Args:
            cam_list (list of dictionaries): List of dictionaries describing initial properties of the views we will
                be setting up. The following is a list of string keys whose values can be defined in those dictionaries:

                capture_height (int): height of video input stream that will be used by this view. If larger than input
                    stream, only entire stream height will be captured. Defaults to entire stream height.

                capture_width (int): width of video input stream that will be used by this view. If larger than input
                    stream, only entire stream width will be captured. Defaults to entire stream width.

                output_height (int): height of video output stream. Defaults to capture_height.

                output_width (int): width of video output stream. Defaults to capture_width.

                center_x (float): float between 0 and 1 representing view center x value. Defaults to 0.5.

                center_y (float): float between 0 and 1 representing view center x value. Defaults to 0.5.

                input_source (string): String specifying where input is coming from. Defaults to 'desktop-rtp-stream',
                    unless capture is smaller than 800x600, in which it defaults to 'desktop-pipe-stream'.

                stream_number (int): If an input_source returns multiple streams, select this one. Defaults to 0.

        Note: This function was hacked together using much trial and error. Functions that should've worked didn't.
            please report any errors your system experiences on the github page so I can fix them.

        Sources:
            Zulko. (n.d.). Read and write video frames in Python using FFMPEG.
                Retrieved December 23, 2016,
                from http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
            Pearson, A. (2008). Subprocess Hanging: PIPE is your enemy.
                Retrieved December 23, 2016,
                from https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
            Documentation. (n.d.).
                Retrieved December 23, 2016,
                from https://ffmpeg.org/documentation.html
         """

        # Set Defaults:

        self.cam_list, self.sources = self._populate_cam_list(cam_list)


        self.transfer_methods_used = []
        if any('desktop-pipe-stream' in substring for substring in self.sources):
            self.transfer_methods_used.append('pipe-file')
        if any('desktop-rtp-stream' in substring for substring in self.sources):
            self.transfer_methods_used.append('rtp')
        if any('desktop-rtp-stream' or 'desktop-pipe-stream' in substring for substring in self.sources):
            self.transfer_methods_used.append('ffmpeg')

        if os.name == 'nt': # if windows
            # Todo: check gdigrab functionality again.

            # Cleanup:
            # We have to kill any other ffmpeg processes, otherwise streaming errors happen.
            os.system('taskkill /F /im ffmpeg.exe')
            # Todo: pop-up warning with "don't show again" box.

            # Setup:
            command = self._cam_to_ffmpeg(cam_list, self.sources, compress=True)

            if 'pipe-file' in self.transfer_methods_used:
                self.temporary_file = tf.NamedTemporaryFile(mode='w+b', buffering=0, delete=False)
                self.frame_number = 0
            # Start:
                self.p = sp.Popen(command, stdout=self.temporary_file, stdin=sp.PIPE)
            else:
                self.p = sp.Popen(command, stdout=sp.PIPE, stdin=sp.PIPE)

            if 'rtp' in self.transfer_methods_used:
                if not self.cv_cam.set(cv2.CAP_PROP_BUFFERSIZE, 3):
                    warnings.warn("Could not set OpenCV buffer size.", RuntimeWarning)
                if not self.cv_cam.open("stream.sdp"):
                    raise cv2.error("OpenCV could not open "+"stream.sdp"+ " file.")
        else:
            # This isn't really what this error is meant for, but it's close enough.
            raise NotImplementedError("Your system, '"+str(os.name)+"' isn't officially supported yet. "
                                                                    "Please send this system name to "
                                                                    "https://github.com/SimLeek/pyoms/issues")

    def end_cast(self):
        self.p.communicate(b'q\n')

    def terminate_cast(self):
        self.p.terminate()
        if 'pipe-file' in self.transfer_methods_used:
            self.cv_cam.release()
        if 'rtp' in self.transfer_methods_used:
            self.temporary_file.close()

    @property
    def get_image(self):
        #
        # Important! If other algorithms stop this from being executed often, output will be buggy
        # currently does not work with point fields. Use texture mapping.
        # Todo: consider putting this read method on its own thread.
        raw_image = self.cv_cam.read()
        raw_image = (True, cv2.cvtColor(raw_image[1], cv2.COLOR_BGR2RGB))
        images = []

        if raw_image[0] is False:
            return [(False, raw_image)]

        height, width, channels = raw_image[1].shape

        for i in range(len(self.cam_list) - 1):
            top_left_x = int(max(self.cam_list[i]['center_x'] - self.cam_list[i]['capture_width'] / 2, 0))
            top_left_y = int(max(self.cam_list[i]['center_y'] - self.cam_list[i]['capture_height'] / 2, 0))

            bottom_right_x = int(min(width, top_left_x + int(self.cam_list[i]['capture_width'])))
            bottom_right_y = int(min(height, top_left_y + int(self.cam_list[i]['capture_height'])))

            cropped_image = raw_image[1][
                            top_left_y:bottom_right_y,
                            top_left_x:bottom_right_x
                            ]
            shrunken_image = cv2.resize(cropped_image, (self.cam_list[i]['output_width'],
                                                        self.cam_list[i]['output_height']),
                                        interpolation=cv2.INTER_AREA)
            # via: http://stackoverflow.com/a/32455269
            gaussian_3 = cv2.GaussianBlur(shrunken_image, (15, 15), 20.0)
            unsharp_image = cv2.addWeighted(shrunken_image, 1.5, gaussian_3, -0.5, 0)
            images.append((True, unsharp_image))

        self.temporary_file.seek(self.frame_number * self.blink_area_multiplier * self.blink_area_multiplier * self.cam_list[-1][
            'capture_width'] * self.cam_list[-1]['capture_height'] * 3)
        raw_image = self.temporary_file.read(
            self.blink_area_multiplier * self.blink_area_multiplier * self.cam_list[-1]['capture_width'] *
            self.cam_list[-1]['capture_height'] * 3)

        # make sure we don't try to seek past the end of the file:
        if raw_image is None:
            self.frame_number -= 1
            images.append((False, None))
            return images
        if len(raw_image) < self.blink_area_multiplier * self.blink_area_multiplier * self.cam_list[-1][
                                            'capture_width'] * self.cam_list[-1]['capture_height'] * 3:
            self.frame_number -= 1
            images.append((False, None))
            return images

        try:
            image = np.fromstring(raw_image, dtype='uint8')
            image = image.reshape(
                (int(self.cam_list[-1]['capture_height'] * self.blink_area_multiplier),
                 int(self.cam_list[-1]['capture_width'] * self.blink_area_multiplier),
                 3)
            )

            '''top_left_x = int(max(self.cam_list[-1]['center_x'] - self.cam_list[-1]['capture_width'] / 2, 0))
            top_left_y = int(max(self.cam_list[-1]['center_y'] - self.cam_list[-1]['capture_height'] / 2, 0))

            bottom_right_x = int(min(width, top_left_x + int(self.cam_list[i]['capture_width'])))
            bottom_right_y = int(min(height, top_left_y + int(self.cam_list[i]['capture_height'])))

            cropped_image = raw_image[1][
                            top_left_y:bottom_right_y,
                            top_left_x:bottom_right_x
                            ]'''
            gaussian_3 = cv2.GaussianBlur(image, (15, 15), 20.0)
            unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0)

            self.frame_number += 1
            # print(self.frame_number)
            # self.p.stdout.flush()
            self.temporary_file.flush()
            images.append((True, unsharp_image))
        except ValueError:
            return images

        return images
