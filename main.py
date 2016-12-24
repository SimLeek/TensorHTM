import htm_vis as htmv
import desktop_io as dio
from vtk.util import numpy_support
import vtk
import numpy as np

g_capture_height = 1080 / 2
g_capture_width = 1920 / 2
g_output_height = 60
g_output_width = 120


class VidPointLoop(htmv.TimerCallback):
    def __init__(self):
        super().__init__()


    def start(self):
        self.sc = dio.OculomotorSystem()
        self.sc.start_cast([{'center_x': 0.5, 'center_y': 0.5,
                              'capture_height': g_capture_height, 'capture_width': g_capture_width,
                              'output_height': g_output_height, 'output_width': g_output_width},
                            {'center_x': 0.5, 'center_y': 0.5,
                              'capture_height': g_capture_height / 6, 'capture_width': g_capture_width / 6,
                              'output_height': g_output_height, 'output_width': g_output_width},
                            {'center_x': 0.5, 'center_y': 0.5,  # actual center
                              'capture_height': g_output_height, 'capture_width': g_output_width,
                              'output_height': g_output_height, 'output_width': g_output_width}
                            ])

    def loop(self, obj, event):
        img_list = self.sc.get_image
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        color_list = []
        for j in range(len(img_list)):
            if img_list[j][0]:
                img = img_list[j][1]
                img_shape = img.shape
                img = np.swapaxes(img, 0, 1)
                new_img = img.reshape(img_shape[0] * img_shape[1], 3, order='C')
                if j == 0:
                    color_list = new_img
                else:
                    color_list = np.concatenate((color_list, new_img))
            else:
                return
        #todo: move to htm_vis
        vtk_data = numpy_support.numpy_to_vtk(num_array=color_list, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        self.point_colors.DeepCopy(vtk_data)

        '''self.point_colors.SetTypedTuple(i, [int(i%255),
                                                    int((i/255)%255),
                                                    int(((i/255)%255)%255)])'''

    def end(self):
        self.sc.end_cast()


if __name__ == "__main__":
    point_displayer = htmv.PointDisplayer(VidPointLoop)

    htmv.add_array(point_displayer, [g_output_height, g_output_width, 3], [0, 1, 0], [0, 0, 0],
                   [int(255 * .2), int(255 * .1), int(255 * .0)])

    point_displayer.set_poly_data()

    point_displayer.visualize()
