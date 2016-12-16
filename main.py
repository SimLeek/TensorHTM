import HTMVis as htmv
import desktop_io as dio
import SimplePipe as sp
import numpy as np

g_height = 120
g_width = 160

class VidPointLoop(htmv.TimerCallback):
    def __init__(self):
        super().__init__()

    def start(self):
        self.sc = dio.ScreenCaster()
        self.sc.cast_screen(height= g_height, width= g_width)

    def loop(self, obj, event):
        img = self.sc.get_image()
        if img is not False:
            for i in range(0, g_height*g_width):
                self.point_colors.SetTypedTuple(i, [int(img[int(i % g_height)][int((i / g_height))][0]),
                                                    int(img[int(i % g_height)][int((i / g_height))][1]),
                                                    int(img[int(i % g_height)][int((i / g_height))][2])])
                '''self.point_colors.SetTypedTuple(i, [int(i%255),
                                                    int((i/255)%255),
                                                    int(((i/255)%255)%255)])'''

    def end(self):
        self.sc.end_cast()

if __name__ == "__main__":
    point_displayer = htmv.PointDisplayer(VidPointLoop)

    htmv.add_array(point_displayer, [g_height, g_width, 1], [0, 1, 0], [0, 0, 0], [int(255 * .2), int(255 * .1), int(255 * .0)])

    point_displayer.set_poly_data()

    point_displayer.visualize()