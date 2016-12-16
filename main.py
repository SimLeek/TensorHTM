import HTMVis as htmv
import desktop_io as dio
import SimplePipe as sp
import numpy as np

#todo: fix non square aspect ratio problem
g_width = 120
g_height = 120

class VidPointLoop(htmv.TimerCallback):
    def __init__(self):
        super().__init__()

    def start(self):
        self.sc = dio.ScreenCaster()
        self.sc.cast_screen(width = g_width, height = g_height)

    def loop(self, obj, event):
        img = self.sc.get_image()
        if img is not False:
            for i in range(0, g_width*g_height):
                self.point_colors.SetTypedTuple(i, [int(img[int(i % g_width)][int((i/g_width))][0]),
                                                    int(img[int(i % g_width)][int((i/g_width))][1]),
                                                    int(img[int(i % g_width)][int((i/g_width))][2])])
                '''self.point_colors.SetTypedTuple(i, [int(i%255),
                                                    int((i/255)%255),
                                                    int(((i/255)%255)%255)])'''

    def end(self):
        self.sc.end_cast()

if __name__ == "__main__":
    point_displayer = htmv.PointDisplayer(VidPointLoop)

    htmv.add_array(point_displayer, [g_width, g_height, 1], [0,1,0], [0,0,0], [int(255 * .2), int(255 * .1), int(255 * .0)])

    point_displayer.set_poly_data()

    point_displayer.visualize()