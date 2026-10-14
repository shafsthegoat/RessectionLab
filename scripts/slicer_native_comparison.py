"""Run using official Slicer --testing --python-script; same-data visual reference."""

import json, os
from pathlib import Path
import numpy as np
import vtk
import slicer
import qt

root = Path(os.environ["RESECTIONLAB_NATIVE_COLLECTION"])
out = Path(os.environ["RESECTIONLAB_COMPARISON_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
doc = json.loads((root / "patient-anatomy.json").read_text())
source = next(s for s in doc["series"] if s["id"] == "t1ce")
volume = slicer.util.loadVolume(str(root / source["original"]["path"]))
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
view = slicer.app.layoutManager().threeDWidget(0).threeDView()
view.renderWindow().SetSize(1100, 900)
segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
segmentation.CreateDefaultDisplayNodes()
segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
for name, color in [("vessels", (0.94, 0.34, 0.47)), ("tumor", (0.77, 0.43, 0.95))]:
    layer = next(x for x in doc["layers"] if x["id"] == name)
    array = np.load(root / layer["data"]["path"])
    mask = (array == layer["label_value"]).astype(np.uint8)
    label = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    # Slicer's NIfTI reader may reverse an axis to obtain a right-handed voxel
    # grid. Our arrays retain the original voxel ordering, so they require their
    # own declared affine rather than copying the reader's reordered geometry.
    matrix=vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i,j,layer['affine'][i][j])
    label.SetIJKToRASMatrix(matrix)
    slicer.util.updateVolumeFromArray(label, mask.transpose(2, 1, 0))
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        label, segmentation
    )
    segmentId = segmentation.GetSegmentation().GetNthSegmentID(
        segmentation.GetSegmentation().GetNumberOfSegments() - 1
    )
    segmentation.GetSegmentation().GetSegment(segmentId).SetName(name)
    segmentation.GetSegmentation().GetSegment(segmentId).SetColor(color)
    slicer.mrmlScene.RemoveNode(label)
segmentation.CreateClosedSurfaceRepresentation()
logic = slicer.modules.volumerendering.logic()
display = logic.CreateDefaultVolumeRenderingNodes(volume)
node = display.GetVolumePropertyNode()
preset = logic.GetPresetByName("MR-Default")
if preset:
    node.Copy(preset)
view.resetFocalPoint()
camera = (
    slicer.modules.cameras.logic()
    .GetViewActiveCameraNode(view.mrmlViewNode())
    .GetCamera()
)
report = json.loads((out / "report.json").read_text())
state = report.get("camera")
if state:
    camera.SetPosition(*state["position"])
    camera.SetFocalPoint(*state["target"])
    camera.SetViewUp(*state["up"])
    camera.SetViewAngle(34)
    camera.SetClippingRange(.1,5000)
view.mrmlViewNode().SetBackgroundColor(0.02, 0.03, 0.05)
view.mrmlViewNode().SetBackgroundColor2(0.02, 0.03, 0.05)
view.mrmlViewNode().SetBoxVisible(False)
view.mrmlViewNode().SetAxisLabelsVisible(False)


def capture(name):
    view.forceRender()
    image = vtk.vtkWindowToImageFilter()
    image.SetInput(view.renderWindow())
    image.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(out / name))
    writer.SetInputConnection(image.GetOutputPort())
    writer.Write()


def finish():
    try:
        display.SetVisibility(False)
        capture("slicer-native-surfaces.png")
        display.SetVisibility(True)
        slicer.app.processEvents()
        capture("slicer-native-volume.png")
        receipt = {
            "slicer_version": slicer.app.applicationVersion,
            "slicer_revision": slicer.app.repositoryRevision,
            "snapshot_id": doc["snapshot_id"],
            "source_sha256": source["original"]["sha256"],
            "source_shape": source["data"]["shape"],
            "surface_method": "Slicer default closed surface conversion of native binary labels",
            "volume_preset": "MR-Default",
            "camera": state,
            "comparison_scope": "same source and masks; transfer functions differ; visual reference only, no accuracy inference",
            "label_geometry":"each native label has its own explicit voxel-to-RAS affine; no assumption about Slicer reader voxel ordering",
        }
        (out / "slicer-reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
    except Exception as e:
        (out / "slicer-error.txt").write_text(repr(e))
        raise
    finally:
        slicer.app.exit()


# --testing may terminate after the startup script returns. Process the render
# queue synchronously, so a successful process always has an image receipt.
for _ in range(20):
    slicer.app.processEvents()
finish()
