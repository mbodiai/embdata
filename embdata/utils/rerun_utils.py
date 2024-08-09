from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from embdata.motion.control import Pose
from embdata.sense.camera import CameraParams

if TYPE_CHECKING:
    from embdata.ndarray import NumpyArray


def get_blueprint() -> rrb.Blueprint:

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="Scene",
                    background=[0.0, 0.0, 0.0, 0.0],
                    origin="scene",
                    visible=True,
                ),
                rrb.Spatial2DView(
                    name="Augmented",
                    background=[0.0, 0.0, 0.0, 0.0],
                    origin="augmented",
                    visible=True,
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Actions",
                    origin="action",
                    visible=True,
                    axis_y=rrb.ScalarAxis(range=(-0.5, 0.5), zoom_lock=True),
                    plot_legend=rrb.PlotLegend(visible=True),
                    time_ranges=[rrb.VisibleTimeRange("timeline0", start=rrb.TimeRangeBoundary.cursor_relative(seq=-100), end=rrb.TimeRangeBoundary.cursor_relative())],
                ),
            ),
            row_shares=[2, 1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )

def log_scalar(name: str, value: float) -> None:
    rr.log(name, rr.Scalar(value))

def project_points_to_2d(camera_params: CameraParams, start_pose: Pose, end_pose: Pose) -> Tuple[np.ndarray, np.ndarray]:

    intrinsic = camera_params.intrinsic.matrix
    distortion = np.array([camera_params.distortion.k1, camera_params.distortion.k2, camera_params.distortion.p1, camera_params.distortion.p2, camera_params.distortion.k3]).reshape(5, 1)

    translation = np.array(camera_params.extrinsic.translation_vector).reshape(3, 1)
    rotation = cv2.Rodrigues(np.array(camera_params.extrinsic.rotation_vector).reshape(3, 1))[0]
    end_effector_offset = 0.175

    # Switch x and z coordinates for the 3D points
    start_position_3d: NumpyArray[3, 1] = np.array([start_pose.z - end_effector_offset, -start_pose.y, start_pose.x]).reshape(3, 1)
    end_position_3d: NumpyArray[3, 1] = np.array([end_pose.z - end_effector_offset, -end_pose.y, end_pose.x]).reshape(3, 1)

    # Transform the 3D point to the camera frame
    start_position_3d_camera_frame: NumpyArray[3, 1] = np.dot(rotation, start_position_3d) + translation
    end_position_3d_camera_frame: NumpyArray[3, 1] = np.dot(rotation, end_position_3d) + translation

    # Project the transformed 3D point to 2D
    start_point_2d, _ = cv2.projectPoints(objectPoints=start_position_3d_camera_frame,
                                            rvec=np.zeros((3,1)),
                                            tvec=np.zeros((3,1)),
                                            cameraMatrix=intrinsic,
                                            distCoeffs=distortion)

    end_point_2d, _ = cv2.projectPoints(objectPoints=end_position_3d_camera_frame,
                                            rvec=np.zeros((3,1)),
                                            tvec=np.zeros((3,1)),
                                            cameraMatrix=intrinsic,
                                            distCoeffs=distortion)

    return start_point_2d[0][0], end_point_2d[0][0]
