import HTMVis as htmv
import desktop_io as dio
import socket as sock
import numpy as np

g_capture_height = 1080
g_capture_width = 1920
g_output_height = 34
g_output_width = 60

class VidPointLoop(htmv.TimerCallback):
    def __init__(self):
        super().__init__()

    def start(self):
        self.sc = dio.ScreenCaster()
        self.sc.cast_screen([{'center_x':1920/2, 'center_y':1080/2,
                            'capture_height': g_capture_height, 'capture_width':g_capture_width,
                            'output_height':g_output_height,'output_width':g_output_width},
                             {'center_x': 1920 / 2, 'center_y': 1080 / 2,
                              'capture_height': g_capture_height/6, 'capture_width': g_capture_width/6,
                              'output_height': g_output_height, 'output_width': g_output_width},
                             {'center_x': 1920 / 2, 'center_y': 1080 / 2,
                              'capture_height': g_capture_height/24, 'capture_width': g_capture_width/24,
                              'output_height': g_output_height, 'output_width': g_output_width}
                            ])


    def loop(self, obj, event):
        img_list = self.sc.get_image()
        for j in range(len(img_list)):
            if img_list[j] != [] and img_list[j] != False:
                print(img_list[j])
                for i in range(g_output_height*g_output_width*j, g_output_height*g_output_width*(j+1)):
                    self.point_colors.SetTypedTuple(i, [int(img_list[j][int(i % g_output_height)][int(((i-g_output_height*g_output_width*j) / g_output_height))][0]),
                                                        int(img_list[j][int(i % g_output_height)][int(((i-g_output_height*g_output_width*j) / g_output_height))][1]),
                                                        int(img_list[j][int(i % g_output_height)][int(((i-g_output_height*g_output_width*j) / g_output_height))][2])])


        '''self.point_colors.SetTypedTuple(i, [int(i%255),
                                                    int((i/255)%255),
                                                    int(((i/255)%255)%255)])'''

    def end(self):
        self.sc.end_cast()

if __name__ == "__main__":
    point_displayer = htmv.PointDisplayer(VidPointLoop)

    htmv.add_array(point_displayer, [g_output_height, g_output_width, 3], [0, 1, 0], [0, 0, 0], [int(255 * .2), int(255 * .1), int(255 * .0)])

    point_displayer.set_poly_data()

    point_displayer.visualize()