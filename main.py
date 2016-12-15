import HTMVis as htmv
import desktop_io as dio
import SimplePipe as sp
import numpy as np

class VidPointLoop(htmv.TimerCallback):
    def __init__(self):
        super().__init__()

    def start(self):
        self.sc = dio.ScreenCaster()
        self.sc.cast_screen(width = 100, height = 100)

    def loop(self, obj, event):
        img = self.sc.get_image()
        for i in range(0, 100*100):
            self.point_colors.SetTypedTuple(i, [int(img[int(i%100)][int(i/100)][0]),
                                                int(img[int(i % 100)][int(i / 100)][1]),
                                                int(img[int(i % 100)][int(i / 100)][2])])

    def end(self):
        self.sc.end_cast()

if __name__ == "__main__":
    point_displayer = htmv.PointDisplayer(VidPointLoop)

    htmv.add_array(point_displayer, [100, 100, 1], [0,1,0], [0,0,0], [int(255 * .2), int(255 * .1), int(255 * .0)])

    point_displayer.set_poly_data()

    point_displayer.visualize()