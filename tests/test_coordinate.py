# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from embdata.geometry import CoordinateField, Pose6D
import numpy as np

# Unit Tests
import unittest


class TestPose6D(unittest.TestCase):
    def test_angular_conversion(self):
        class PoseRollDegrees(Pose6D):
            roll: float = CoordinateField(unit="deg")

        pose = PoseRollDegrees(roll=45.0)
        pose_in_radians: Pose6D = pose.to(angular_unit="rad")
        self.assertAlmostEqual(pose_in_radians.roll, np.pi / 4)

    def test_linear_conversion(self):
        class PoseMM(Pose6D):
            x: float = CoordinateField(unit="mm")
            y: float = CoordinateField(unit="mm")
            z: float = CoordinateField(unit="mm")

        pose = PoseMM(x=1000.0, y=0.0, z=1.0)
        pose_in_meters: PoseMM = pose.to(unit="m")
        self.assertEqual(pose_in_meters.x, 1.0)
        self.assertEqual(pose_in_meters.y, 0.0)
        self.assertEqual(pose_in_meters.z, 0.001)

        pose_in_centimeters = pose.to(unit="cm")
        self.assertEqual(pose_in_centimeters.x, 100.0)
        self.assertEqual(pose_in_centimeters.y, 0.0)
        self.assertEqual(pose_in_centimeters.z, 0.1)


if __name__ == "__main__":
    unittest.main()
