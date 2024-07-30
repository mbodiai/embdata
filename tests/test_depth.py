import numpy as np
from PIL import Image
import pytest
from embdata.sense.depth import Depth

def test_depth_initialization():
  depth = Depth(mode="I", points=None, array=None)
  assert depth.mode == "I"
  assert depth.points is None
  assert depth.array is None

def test_depth_from_pil():
  pil_image = Image.new("RGB", (100, 100))
  depth = Depth.from_pil(pil_image)
  assert depth.mode == "I"
  assert depth.points is None
  assert depth.array is not None
  assert isinstance(depth.array, np.ndarray)

def test_depth_cluster_points():
  depth = Depth()
  depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  labels = depth.cluster_points(n_clusters=2)
  assert len(labels) == 3
  assert set(labels) == {0, 1}

def test_depth_segment_plane():
  depth = Depth()
  depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  inlier_mask, plane_coefficients = depth.segment_plane()
  assert inlier_mask.shape == (3,)
  assert plane_coefficients.shape == (4,)

def test_depth_show():
  depth = Depth()
  depth.array = np.zeros((100, 100, 3), dtype=np.uint8)
  depth.show()  # Just checking if the function runs without errors

def test_depth_segment_cylinder():
  depth = Depth()
  depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  inlier_mask, cylinder_coefficients = depth.segment_cylinder()
  assert inlier_mask.shape == (3,)
  assert cylinder_coefficients.shape == (3,)
