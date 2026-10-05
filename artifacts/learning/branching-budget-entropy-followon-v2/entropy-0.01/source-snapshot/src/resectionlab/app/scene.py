"""VTK anatomy and complete-tool display in the same RAS millimeter frame."""

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkImageActor, vtkWindowToImageFilter
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 -- registers native rendering implementation
import vtkmodules.vtkInteractionStyle  # noqa: F401 -- registers trackball interaction
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def mask_surface(mask):
    """Build a voxel-coordinate mesh; the source affine is applied at display."""
    image = vtkImageData()
    image.SetDimensions(*mask.shape)
    image.GetPointData().SetScalars(numpy_to_vtk(np.asarray(mask, dtype=np.uint8).ravel(order="F"), deep=True))
    surface = vtkFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetValue(0, .5)
    surface.ComputeNormalsOn()
    surface.Update()
    poly = vtkPolyData()
    poly.DeepCopy(surface.GetOutput())
    return poly


class AnatomyScene(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        self._needs_paint = True
        super().__init__(parent)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(.024, .035, .043)
        self.renderer.SetBackground2(.061, .086, .098)
        self.renderer.GradientBackgroundOn()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.GetRenderWindow().SetMultiSamples(0)
        self.actors = {}
        self.route_actors = []
        self.cursor_actor = None
        self.plane_actor = None
        self.replay_actor = None
        self.orientation = None
        self.setMinimumSize(280, 340)
        self.setAccessibleName("3-D anatomy in RAS millimeters")

    def set_surfaces(self, surfaces, affine, colors):
        self.renderer.RemoveAllViewProps()
        self.actors.clear()
        self.route_actors.clear()
        self.cursor_actor = self.plane_actor = self.replay_actor = None
        matrix = vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                matrix.SetElement(i, j, float(affine[i, j]))
        for name, mesh in surfaces.items():
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.SetUserMatrix(matrix)
            color = colors.get(name, (125, 147, 155))
            actor.GetProperty().SetColor(*[c / 255 for c in color])
            actor.GetProperty().SetOpacity(.21 if name == "brain" else .76)
            actor.GetProperty().SetAmbient(.28)
            actor.GetProperty().SetDiffuse(.72)
            actor.GetProperty().SetSpecular(.22)
            actor.GetProperty().SetSpecularPower(25)
            self.actors[name] = actor
            self.renderer.AddActor(actor)
        self.reset_camera()

    def show_orientation(self):
        axes = vtkAxesActor()
        axes.SetXAxisLabelText("R")
        axes.SetYAxisLabelText("A")
        axes.SetZAxisLabelText("S")
        axes.SetTotalLength(1., 1., 1.)
        marker = vtkOrientationMarkerWidget()
        marker.SetOrientationMarker(axes)
        marker.SetInteractor(self.GetRenderWindow().GetInteractor())
        marker.SetViewport(0., 0., .23, .23)
        marker.SetEnabled(True)
        marker.InteractiveOff()
        self.orientation = marker

    def set_replay_surface(self, mask, affine):
        if self.replay_actor is not None:
            self.renderer.RemoveActor(self.replay_actor)
            self.replay_actor = None
        if mask is not None and np.any(mask):
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(mask_surface(mask))
            actor = vtkActor()
            actor.SetMapper(mapper)
            matrix = vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    matrix.SetElement(i, j, float(affine[i, j]))
            actor.SetUserMatrix(matrix)
            actor.GetProperty().SetColor(.37, .76, .94)
            actor.GetProperty().SetOpacity(.72)
            actor.GetProperty().SetEdgeVisibility(True)
            actor.GetProperty().SetEdgeColor(.15, .35, .45)
            self.renderer.AddActor(actor)
            self.replay_actor = actor
        self.render()

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(300, -400, 220)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        camera.Zoom(1.2)
        self.render()

    def set_visibility(self, name, visible):
        if name in self.actors:
            self.actors[name].SetVisibility(visible)
            self.render()

    def set_cursor(self, point):
        if self.cursor_actor is None:
            sphere = vtkSphereSource()
            sphere.SetRadius(1.2)
            sphere.SetThetaResolution(16)
            sphere.SetPhiResolution(16)
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            self.cursor_actor = vtkActor()
            self.cursor_actor.SetMapper(mapper)
            self.cursor_actor.GetProperty().SetColor(.63, .91, .83)
            self.renderer.AddActor(self.cursor_actor)
        self.cursor_actor.SetPosition(*point)

    def set_axial_image(self, rgb, geometry, visible=True):
        image = vtkImageData()
        image.SetDimensions(geometry.width, geometry.height, 1)
        dx, dy = geometry.physical_size
        image.SetSpacing(dx / (geometry.width - 1), dy / (geometry.height - 1), 1)
        image.SetOrigin(geometry.bounds[0, 0], geometry.bounds[0, 1], geometry.position_mm)
        scalars = numpy_to_vtk(np.flipud(rgb).reshape(-1, 3).copy(), deep=True)
        image.GetPointData().SetScalars(scalars)
        if self.plane_actor is None:
            self.plane_actor = vtkImageActor()
            self.renderer.AddActor(self.plane_actor)
        self.plane_actor.SetInputData(image)
        self.plane_actor.SetVisibility(visible)
        self.plane_actor.GetProperty().SetOpacity(.9)

    def set_routes(self, routes):
        for actor in self.route_actors:
            self.renderer.RemoveActor(actor)
        self.route_actors.clear()
        palette = [(.51, .87, .74), (.65, .69, .96)]
        for index, (entry, target, radius, length, rejected, failure_point) in enumerate(routes):
            entry, target = np.asarray(entry), np.asarray(target)
            direction = target - entry
            direction /= max(np.linalg.norm(direction), 1e-12)
            # Show the complete rigid shaft, including its extracranial portion.
            proximal = target - direction * length
            points = vtkPoints()
            points.InsertNextPoint(*proximal)
            points.InsertNextPoint(*target)
            lines = vtkCellArray()
            lines.InsertNextCell(2)
            lines.InsertCellPoint(0)
            lines.InsertCellPoint(1)
            poly = vtkPolyData()
            poly.SetPoints(points)
            poly.SetLines(lines)
            tube = vtkTubeFilter()
            tube.SetInputData(poly)
            tube.SetRadius(radius)
            tube.SetNumberOfSides(20)
            tube.CappingOn()
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*((.9, .38, .32) if rejected else palette[index % 2]))
            actor.GetProperty().SetOpacity(.85)
            self.renderer.AddActor(actor)
            self.route_actors.append(actor)
            if failure_point is not None:
                sphere = vtkSphereSource()
                sphere.SetCenter(*failure_point)
                sphere.SetRadius(max(radius * 1.6, 2))
                sphere_mapper = vtkPolyDataMapper()
                sphere_mapper.SetInputConnection(sphere.GetOutputPort())
                failure_actor = vtkActor()
                failure_actor.SetMapper(sphere_mapper)
                failure_actor.GetProperty().SetColor(.98, .32, .23)
                self.renderer.AddActor(failure_actor)
                self.route_actors.append(failure_actor)
        self.render()

    def render(self):
        if self.isVisible():
            self.GetRenderWindow().Render()

    def capture_rgb(self):
        """Read the native OpenGL framebuffer; Qt grabs omit native children."""
        self.render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(self.GetRenderWindow())
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        result = capture.GetOutput()
        width, height, _ = result.GetDimensions()
        pixels = vtk_to_numpy(result.GetPointData().GetScalars()).reshape(height, width, 3)
        return np.flipud(pixels).copy()

    def paintEvent(self, event):
        # Cocoa's native OpenGL child sends expose events after buffer swaps.
        # Rendering unconditionally for every expose can starve Qt timers and
        # worker signals. Scene edits/interactions already request a render;
        # only a genuinely invalidated widget needs this extra paint.
        if self._needs_paint:
            self._needs_paint = False
            super().paintEvent(event)

    def resizeEvent(self, event):
        self._needs_paint = True
        super().resizeEvent(event)
