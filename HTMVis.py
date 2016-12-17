#import tensorflow as tf
#import numpy as np
#import vtk

# !/usr/bin/env python

#import HTMNeuron
import vtk
import math as m
import time
import bisect
import random
import numpy as np

global_keyDic = None

global_interactor_parent = None

global_camera = None
global_camera_renderWindow = None

def clamp(n, minn, maxn):
    # http://stackoverflow.com/a/5996949
    return max(min(maxn, n), minn)

class TimerCallback(object):

    __slots__ = ["points", "point_colors", "timer_count", "points_poly", "last_velocity_update", "unused_locations", "last_color_velocity_update", "renderer", "last_bgcolor_velocity_update"]

    def del_points(self, point_indices):
        if isinstance(point_indices, (tuple, list)):
            for i in range(len(point_indices)):
                # move point to cornfield so nobody sees it
                #todo:remove from displayer
                self.points.SetPoint(point_indices[i], (float("nan"),float("nan"),float("nan")))
                # self.points.SetPoint(point_indices[i], (float("-inf"),float("-inf"),float("-inf")))
                # self.points.SetPoint(point_indices[i], (float("inf"),float("inf"),float("inf")))

                bisect.insort_right(self.unused_locations, point_indices[i])
        else:
            bisect.insort_right(self.unused_locations, point_indices)
        self.points_poly.Modified()

    def add_points(self, points, point_indices=None):
        #todo: keep array of no longer used point locations
        if isinstance(points[0], (list, tuple)):
            if point_indices is not None:
                for i in range(len(points)):
                    self.points.InsertPoint(point_indices[i], points[i])
            else:
                for i in range(len(points)):
                    if len(self.unused_locations) >0:
                        self.points.SetPoint(self.unused_locations.pop(0), points[i])
                    else:
                        self.points.InsertNextPoint(points[i])
        else:
            if len(self.unused_locations) > 0:
                self.points.SetPoint(self.unused_locations.pop(0), points)
            else:
                self.points.InsertNextPoint(points)
        self.points_poly.Modified()

    def set_bg_color(self, color):
        r,g,b = color
        self.renderer.SetBackground((clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)))
        self.renderer.Modified()

    def move_bg_color(self, color):
        r, g, b = color
        r0, b0, g0 = self.renderer.GetBackground()
        self.renderer.SetBackground((clamp(r+r0,0,1),clamp(g+g0,0,1),clamp(b+b0,0,1)))
        self.renderer.Modified()

    def apply_velocity_to_bg_color(self, color):
        t = time.clock() - self.last_color_velocity_update
        r, g, b = color
        r0, b0, g0 = self.renderer.GetBackground()
        self.renderer.SetBackground((clamp(r*t + r0, 0, 1), clamp(g*t + g0, 0, 1), clamp(b*t + b0, 0, 1)))
        self.renderer.Modified()
        self.last_bgcolor_velocity_update = time.clock()

    def set_point_colors(self, point_indices, positions):
        if isinstance(point_indices, (list, tuple)):
            if isinstance(positions, (list, tuple)):
                for i in range(len(point_indices)):
                    r, g, b = positions[i%len(positions)]
                    self.point_colors.SetTypedTuple(i, [int(r%255), int(g%255), int(b%255)])
            else:
                for i in range(len(point_indices)):
                    r, g, b = positions
                    self.point_colors.SetTypedTuple(i, [int(r%255), int(g%255), int(b%255)])
        else:
            r, g, b = positions
            self.point_colors.SetTypedTuple(point_indices, [int(r%255), int(g%255), int(b%255)])
        # self.points_poly.GetPointData().GetScalars().Modified()
        self.points_poly.Modified()

    def move_point_colors(self, point_indices, changes):
        if isinstance(point_indices, (list, tuple)):
            if isinstance(changes, (list, tuple)):
                for i in range(len(point_indices)):
                    x, y, z = self.point_colors.GetTypedTuple(point_indices[i])
                    vx, vy, vz = changes[i%len(changes)]
                    self.points.SetPoint(point_indices[i], (int(x+vx)%255, int(y+vy)%255, int(z+vz)%255))
            else:
                for i in range(len(point_indices)):
                    x, y, z = self.point_colors.GetTypedTuple(point_indices[i])
                    vx, vy, vz = changes
                    self.points.SetPoint(point_indices[i], (int(x+vx)%255, int(y+vy)%255, int(z+vz)%255))
        else:
            x, y, z = self.points.GetPoint(point_indices)
            vx, vy, vz = changes
            self.points.SetPoint(point_indices, (int(x + vx)%255 , int(y + vy)%255, int(z + vz)%255))

        # self.points_poly.GetPointData().GetScalars().Modified()
        self.points_poly.Modified()

    def apply_velocities_to_point_colors(self, point_indices, velocities):
        t = time.clock() - self.last_color_velocity_update
        if isinstance(point_indices, (list, tuple)):
            if isinstance(velocities, (list, tuple)):
                for i in range(len(point_indices)):
                    r, g, b = self.point_colors.GetTypedTuple(point_indices[i])
                    vx, vy, vz = velocities[i%len(velocities)]
                    self.points.SetPoint(point_indices[i], (int(r+vx*t)%255, int(g+vy*t)%255, int(b+vz*t)%255))
            else:
                for i in range(len(point_indices)):
                    r, g, b = self.point_colors.GetTypedTuple(point_indices[i])
                    vx, vy, vz = velocities
                    self.points.SetPoint(point_indices[i], (int(r+vx*t)%255, int(g+vy*t)%255, int(b+vz*t)%255))
        else:
            r, g, b = self.point_colors.GetTypedTuple(point_indices[i])
            vx, vy, vz = velocities
            self.points.SetPoint(point_indices, (int(r + vx * t)%255, int(g + vy * t)%255, int(b + vz * t)%255))

        # self.points_poly.GetPointData().GetScalars().Modified()
        self.points_poly.Modified()
        self.last_color_velocity_update = time.clock()

    def position_points(self, point_indices, positions):
        if isinstance(point_indices, (list, tuple)):
            if isinstance(positions, (list, tuple)):
                for i in range(len(point_indices)):
                    x, y, z = self.points.GetPoint(point_indices[i])
                    vx, vy, vz = positions[i%len(positions)]
                    self.points.SetPoint(point_indices[i], (vx, vy, vz))
            else:
                for i in range(len(point_indices)):
                    x, y, z = self.points.GetPoint(point_indices[i])
                    vx, vy, vz = positions
                    self.points.SetPoint(point_indices[i], (vx, vy, vz))
        else:
            x, y, z = self.points.GetPoint(point_indices)
            vx, vy, vz = positions
            self.points.SetPoint(point_indices, (vx , vy, vz))
        self.points_poly.Modified()

    def move_points(self, point_indices, changes):
        if isinstance(point_indices, (list, tuple)):
            if isinstance(changes, (list, tuple)):
                for i in range(len(point_indices)):
                    x, y, z = self.points.GetPoint(point_indices[i])
                    vx, vy, vz = changes[i%len(changes)]
                    self.points.SetPoint(point_indices[i], (x+vx, y+vy, z+vz))
            else:
                for i in range(len(point_indices)):
                    x, y, z = self.points.GetPoint(point_indices[i])
                    vx, vy, vz = changes
                    self.points.SetPoint(point_indices[i], (x+vx, y+vy, z+vz))
        else:
            x, y, z = self.points.GetPoint(point_indices)
            vx, vy, vz = changes
            self.points.SetPoint(point_indices, (x + vx , y + vy, z + vz))

        self.points_poly.Modified()

    def apply_velocities_to_points(self, point_indices, velocities):
        t = time.clock() - self.last_velocity_update
        if isinstance(point_indices, (list, tuple)):
            if isinstance(velocities, (list, tuple)):
                for i in range(len(point_indices)):
                    x, y, z = self.points.GetPoint(point_indices[i])
                    vx, vy, vz = velocities[i%len(velocities)]
                    self.points.SetPoint(point_indices[i], (x+vx*t, y+vy*t, z+vz*t))
            else:
                for i in range(len(point_indices)):
                    x, y, z = self.points.GetPoint(point_indices[i])
                    vx, vy, vz = velocities
                    self.points.SetPoint(point_indices[i], (x+vx*t, y+vy*t, z+vz*t))
        else:
            x, y, z = self.points.GetPoint(point_indices)
            vx, vy, vz = velocities
            self.points.SetPoint(point_indices, (x + vx * t, y + vy * t, z + vz * t))

        self.points_poly.Modified()
        self.last_velocity_update = time.clock()

    def __init__(self):
        self.timer_count = 0
        self.last_velocity_update = time.clock()
        self.last_color_velocity_update = time.clock()
        self.last_bgcolor_velocity_update = time.clock()
        self.unused_locations = []
        self.start()

    def start(self):
        pass

    def loop(self, obj, event):
        pass

    def end(self):
        pass

    def execute(self, obj, event):
        self.loop(obj, event)
        """for i in range(0, self.points.GetNumberOfPoints()):
            x, y, z = self.points.GetPoint(i)
            self.points.SetPoint(i, (x + random.uniform(-1,1), y+ random.uniform(-1,1), z+ random.uniform(-1,1)))
        for i in range(0, self.point_colors.GetNumberOfTuples(), 3):
            self.point_colors.SetTypedTuple(i, [int((m.sin(self.timer_count / 100) + 1) / 2 * 255),
                                                int((m.sin(self.timer_count / 200) + 1) / 2 * 255),
                                                int((m.sin(self.timer_count / 300) + 1) / 2 * 255)])
        for i in range(1, self.point_colors.GetNumberOfTuples(), 3):
            self.point_colors.SetTypedTuple(i, [int((m.sin(self.timer_count / 150) + 1) / 2 * 255),
                                                int((m.sin(self.timer_count / 250) + 1) / 2 * 255),
                                                int((m.sin(self.timer_count / 350) + 1) / 2 * 255)])
        for i in range(2, self.point_colors.GetNumberOfTuples(), 3):
            self.point_colors.SetTypedTuple(i, [int((m.sin(self.timer_count / 170) + 1) / 2 * 255),
                                                int((m.sin(self.timer_count / 270) + 1) / 2 * 255),
                                                int((m.sin(self.timer_count / 370) + 1) / 2 * 255)])"""

        #print(self.point_colors.GetTuple(i))

        #self.points_poly.GetPointData().SetScalars(self.point_colors)
        self.points_poly.GetPointData().GetScalars().Modified()
        self.points_poly.Modified()

        iren = obj
        iren.GetRenderWindow().Render()

class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, camera, renderWindow, parent = None):
        #should work with else statement, but doesnt for some reason

        global global_interactor_parent
        global_interactor_parent = vtk.vtkRenderWindowInteractor()
        if parent is not None:
            global_interactor_parent = parent

        global global_camera
        global_camera = camera

        global global_camera_renderWindow
        global_camera_renderWindow = renderWindow

        global global_keyDic
        global_keyDic = {
            'w': self._move_forward,
            's': self._move_backward,
            'a': self._yaw_left,
            'd': self._yaw_right,
            'Shift_L': self._pitch_up,
            'space': self._pitch_down
        }

        self.AddObserver("KeyPressEvent", self.keyPress)
        #self.RemoveObservers("LeftButtonPressEvent")
        #self.AddObserver("LeftButtonPressEvent", self.dummy_func)

        #todo: dummy func
        #self.RemoveObservers("RightButtonPressEvent")
        #self.AddObserver("RightButtonPressEvent", self.dummy_func_2)

    def dummy_func(self, obj, ev):
        self.OnLeftButtonDown()

    def dummy_func_2(self, obj, ev):
        pass

    def _move_forward(self):
        #todo: change this to a velocity function with drag and let something else
        # interpolate the velocity over time
        norm = global_camera.GetViewPlaneNormal()
        pos = global_camera.GetPosition()
        global_camera.SetPosition(pos[0] - norm[0]*10,
                                  pos[1] - norm[1]*10,
                                  pos[2] - norm[2]*10)
        global_camera.SetFocalPoint(pos[0] - norm[0] * 20,
                                    pos[1] - norm[1] * 20,
                                    pos[2] - norm[2] * 20)

    def _move_backward(self):
        # todo: change this to a velocity function with drag and let something else
        # interpolate the velocity over time
        norm = global_camera.GetViewPlaneNormal()
        pos = global_camera.GetPosition()
        global_camera.SetPosition(pos[0] + norm[0] * 10,
                                  pos[1] + norm[1] * 10,
                                  pos[2] + norm[2] * 10)
        global_camera.SetFocalPoint(pos[0] - norm[0] * 20,
                                    pos[1] - norm[1] * 20,
                                    pos[2] - norm[2] * 20)

    def _yaw_right(self):
        global_camera.Yaw(-10)
        global_camera_renderWindow.GetInteractor().Render()

    def _yaw_left(self):
        global_camera.Yaw(10)
        global_camera_renderWindow.GetInteractor().Render()

    def _pitch_up(self):
        global_camera.Pitch(10)
        global_camera_renderWindow.GetInteractor().Render()

    def _pitch_down(self):
        global_camera.Pitch(-10)
        global_camera_renderWindow.GetInteractor().Render()


    def keyPress(self, obj, event):
        #self is lost. Gonna have to report this to vtk...
        key = global_interactor_parent.GetKeySym()


        if key in global_keyDic:
            global_keyDic[key]()
        else:
            print(key)


class PointDisplayer:
    #adapted from:
    # http://www.vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/Point
    def __init__(self, callback_class):
        self.points = vtk.vtkPoints()
        self.vertices = vtk.vtkCellArray()

        self.point_colors = vtk.vtkUnsignedCharArray()
        self.point_colors.SetNumberOfComponents(3)
        self.point_colors.SetName("Colors")

        self.lines = vtk.vtkCellArray()

        self.line_colors = vtk.vtkUnsignedCharArray()
        self.line_colors.SetNumberOfComponents(3)
        self.line_colors.SetName("Colors")

        assert issubclass(callback_class, TimerCallback)
        self.callback_class = callback_class


    #def add_point(self, x, y, z):
    #    point = [x,y,z]
    #    self.add_point(point)

    def add_point(self, point, color):
        id = self.points.InsertNextPoint(point)
        self.vertices.InsertNextCell(1)
        self.vertices.InsertCellPoint(id)

        self.point_colors.InsertNextTypedTuple(color)

        return id

    def add_line(self, point_a_index, point_b_index, color):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, point_a_index)
        line.GetPointIds().SetId(1, point_b_index)

        id = self.lines.InsertNextCell(line)

        self.line_colors.InsertNextTypedTuple(color)
        return id

    def set_poly_data(self):
        self.points_poly = vtk.vtkPolyData()
        self.points_poly.SetPoints(self.points)
        self.points_poly.SetVerts(self.vertices)

        self.points_poly.GetPointData().SetScalars(self.point_colors)

        self.lines_poly = vtk.vtkPolyData()
        self.lines_poly.SetPoints(self.points)
        self.lines_poly.SetLines(self.lines)

        self.lines_poly.GetCellData().SetScalars(self.line_colors)

    def visualize(self):
        point_mapper = vtk.vtkPolyDataMapper()
        line_mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            point_mapper.SetInput(self.points_poly)
            line_mapper.SetInput(self.lines_poly)
        else:
            point_mapper.SetInputData(self.points_poly)
            line_mapper.SetInputData(self.lines_poly)

        point_actor = vtk.vtkActor()
        line_actor = vtk.vtkActor()
        point_actor.SetMapper(point_mapper)
        line_actor.SetMapper(line_mapper)
        point_actor.GetProperty().SetPointSize(10)#todo:allow modification
        #actor.GetProperty().SetPointColor

        renderer = vtk.vtkRenderer()

        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetInteractorStyle(
            KeyPressInteractorStyle(camera = renderer.GetActiveCamera(),
                                    renderWindow = renderWindow,
                                    parent = renderWindowInteractor)
        )

        renderWindowInteractor.SetRenderWindow(renderWindow)

        renderer.AddActor(point_actor)
        renderer.AddActor(line_actor)

        #light brown = .6,.6,.4
        # light brown = .2,.2,.1
        #dusk = .05, .05, .1
        #calm blue sky = .1, .2, .4
        #day blue sky = .2, .4, .8
        #bright blue sky = .6, .8, 1.0 (bg attention activation)
        renderer.SetBackground(.6, .8, 1.0) #todo:allow modification

        renderWindow.Render()

        renderWindowInteractor.Initialize()

        cb = self.callback_class()
        cb.renderer = renderer
        cb.points = self.points
        cb.points_poly = self.points_poly
        cb.point_colors = self.point_colors
        renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
        timerId = renderWindowInteractor.CreateRepeatingTimer(30)
        renderWindowInteractor.Start()

        #after loop
        cb.end()

def show_landscape(point_displayer):
    from opensimplex import OpenSimplex
    import random

    simp_r = OpenSimplex(seed=364)
    simp_g = OpenSimplex(seed=535)
    simp_b = OpenSimplex(seed=656)

    for i in range(100000):
        x = random.randint(0, 1000, 4237842 + i)
        y = random.randint(0, 1000, 5437474 + i)

        r1 = .0009765625 * (simp_g.noise2d(x=x, y=y))
        r2 = .001953125 * (simp_r.noise2d(x=x / 2.0, y=y / 2.0))
        r3 = .00390625 * (simp_b.noise2d(x=x / 4.0, y=y / 4.0, ))
        r4 = .0078125 * (simp_g.noise2d(x=x / 8.0, y=y / 8.0))
        r5 = .015625 * (simp_r.noise2d(x=x / 16.0, y=y / 16.0))
        r6 = .03125 * (simp_b.noise2d(x=x / 32.0, y=y / 32.0))
        r7 = .0625 * (simp_g.noise2d(x=x / 64.0, y=y / 64.0))
        r8 = .125 * (simp_r.noise2d(x=x / 128.0, y=y / 128.0))
        r9 = .25 * (simp_b.noise2d(x=x / 256.0, y=y / 256.0))
        r10 = .5 * (simp_g.noise2d(x=x / 512.0, y=y / 512.0))
        r11 = (simp_r.noise2d(x=x / 1024.0, y=y / 1024.0))
        normalization_factor = .5
        val = ((r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9) / 2.0)
        if val > 0:
            p = 1.0
        else:
            p = -1.0
        norm_val = (abs(val) ** normalization_factor) * p
        pos_val = (norm_val + 1.0) / 2.0
        z = pos_val * 254.0

        point_displayer.add_point([x - 100, y - 100, z - 100], [160, int(z), 20])

def show_cloud(point_displayer):
    from opensimplex import OpenSimplex
    import math
    import random

    simp_r = OpenSimplex(seed=364)
    simp_g = OpenSimplex(seed=535)
    simp_b = OpenSimplex(seed=656)

    for i in range(100000):

        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        z = random.randint(0, 1000)

        d = math.sqrt((x - 500) ** 2 + (y - 500) ** 2 + (z - 500) ** 2) / 500.0

        r1 = .0009765625 * (simp_g.noise3d(x=x, y=y, z=z))
        r2 = .001953125 * (simp_r.noise3d(x=x / 2.0, y=y / 2.0, z=z / 2.0))
        r3 = .00390625 * (simp_b.noise3d(x=x / 4.0, y=y / 4.0, z=z / 4.0))
        r4 = .0078125 * (simp_g.noise3d(x=x / 8.0, y=y / 8.0, z=z / 8.0))
        r5 = .015625 * (simp_r.noise3d(x=x / 16.0, y=y / 16.0, z=z / 16.0))
        r6 = .03125 * (simp_b.noise3d(x=x / 32.0, y=y / 32.0, z=z / 32.0))
        r7 = .0625 * (simp_g.noise3d(x=x / 64.0, y=y / 64.0, z=z / 64.0))
        r8 = .125 * (simp_r.noise3d(x=x / 128.0, y=y / 128.0, z=z / 128.0))
        r9 = .25 * (simp_b.noise3d(x=x / 256.0, y=y / 256.0, z=z / 256.0))
        r10 = .5 * (simp_g.noise3d(x=x / 512.0, y=y / 512.0, z=z / 512.0))
        r11 = (simp_r.noise3d(x=x / 1024.0, y=y / 1024.0, z=z / 1024.0))
        normalization_factor = .5
        val = ((r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9) / 2.0)
        if val > 0:
            p = 1.0
        else:
            p = -1.0

        # use ^d for cumulus clouds,
        # use distance from a certain height for a sky of clouds
        # use constant power <1 for endless 3d field of clouds
        # use distance from sets of points or lines for other shapes

        norm_val = (abs(val) ** d) * p
        pos_val = (norm_val + 1.0) / 2.0
        r = int(pos_val * 254.0)
        # r5 = int((r5)*255.0/2.0)
        # lim octaves->inf gives 1/2^x sum (=1)
        if r > 160:
            point_displayer.add_point([x, y, z], [r, r, r])

def show_rand_line_cube(point_displayer):
    import randomSample

    line_a = randomSample.randomSample(xrange(0, 500), 500, 432684)
    line_b = randomSample.randomSample(xrange(500, 1000), 500, 53245643)

    for i in range(len(line_a)):
        r = randomSample.randInt(0, 255, 5453476 + i)
        g = randomSample.randInt(0, 255, 5983279 + i)
        b = randomSample.randInt(0, 255, 9827312 + i)
        point_displayer.add_line(line_a[i], line_b[i], [r, g, b])

def normalize(a, axis=-1, order = 2):
    # from: http://stackoverflow.com/a/21032099
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def add_array(point_displayer, widths, normal, center, color):
    true_normal = normalize(normal)
    if not np.allclose(true_normal, [1,0,0]):
        zn = np.cross(true_normal, [1,0,0])
        xn = np.cross(true_normal, zn)
    else:
        xn = [1,0,0]
        zn = [0,0,1]
    for z in range(-int(m.floor(widths[2] / 2.0)), int(m.ceil(widths[2] / 2.0))):
        for y in range(-int(m.floor(widths[1] / 2.0)), int(m.ceil(widths[1] / 2.0))):
            for x in range(-int(m.floor(widths[0] / 2.0)), int(m.ceil(widths[0] / 2.0))):
                axisizer = np.column_stack((np.transpose(xn),np.transpose(true_normal), np.transpose(zn)))
                translation = np.matmul([x,y,z*10], axisizer)
                point_location = [center[0], center[1], center[2]] + translation
                point_displayer.add_point(point_location, color)

def show_point_field_test(point_displayer):
    from n_d_point_field import n_dimensional_n_split_float
    split_pts = n_dimensional_n_split_float([-200, 200, -150, 150, -175, 175], 83200)
    points = list(split_pts.intersection((-200, -150, -175, 200, 150, 175), objects=True))

    print(split_pts)

    for i in xrange(len(points)):
        x = points[i].bbox[0]
        y = points[i].bbox[1]
        z = points[i].bbox[2]

        col_r = (i % (127)) - 127 * .4
        col_b = (i % (127)) - 127 * .2

        r = int(255 * .4 + col_r)
        g = int(255 * .4 + col_r)
        b = int(255 * .2 + col_b)

        point_displayer.add_point([x, y, z], [r, g, b])



if __name__ == "__main__":
    point_displayer = PointDisplayer()

    # show_cloud(point_displayer)
    add_array(point_displayer, [64, 64, 7], [0,1,0], [0,0,0], [int(255 * .2), int(255 * .1), int(255 * .0)])
    add_array(point_displayer, [64, 64, 7], [0,1,0], [0,0,20], [int(255 * .5), int(255 * .3), int(255 * .1)])
    add_array(point_displayer, [16, 16, 3], [0,1,1], [0,45,0], [int(255 * 0), int(255 * .13), int(255 * 0)])



    point_displayer.set_poly_data()

    point_displayer.visualize()