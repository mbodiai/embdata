import pytest
from embdata.geometry import Coordinate, CoordinateField
from pydantic import ValidationError

def test_create_dynamic_coordinate_objects():
    # Create a 2D point
    point_2d = Coordinate(x=1, y=2)
    assert point_2d.x == 1
    assert point_2d.y == 2

    # Create a 3D point
    point_3d = Coordinate(x=3, y=4, z=5)
    assert point_3d.x == 3
    assert point_3d.y == 4
    assert point_3d.z == 5

    # Create a custom coordinate
    custom_coord = Coordinate(latitude=40.7128, longitude=-74.0060)
    assert custom_coord.latitude == 40.7128
    assert custom_coord.longitude == -74.0060

def test_convert_between_coordinate_types():
    # Convert 2D point to 3D point
    point_2d = Coordinate(x=1, y=2)
    point_3d = Coordinate(x=point_2d.x, y=point_2d.y, z=0)
    assert point_3d.x == 1
    assert point_3d.y == 2
    assert point_3d.z == 0

    # Convert custom coordinate to 2D point
    custom_coord = Coordinate(latitude=40.7128, longitude=-74.0060)
    point_2d = Coordinate(x=custom_coord.longitude, y=custom_coord.latitude)
    assert point_2d.x == -74.0060
    assert point_2d.y == 40.7128

def test_checking_bounds():
    class BoundedCoordinate(Coordinate):
        x = CoordinateField(bounds=(-10, 10))
        y = CoordinateField(bounds=(-10, 10))

    # This should work
    valid_coord = BoundedCoordinate(x=5, y=5)
    assert valid_coord.x == 5
    assert valid_coord.y == 5

    # This should raise a ValidationError
    with pytest.raises(ValidationError):
        invalid_coord = BoundedCoordinate(x=15, y=5)

def test_different_coordinate_types():
    class Point2D(Coordinate):
        x: float
        y: float

    class Point3D(Coordinate):
        x: float
        y: float
        z: float

    class GPSCoordinate(Coordinate):
        latitude: float
        longitude: float

    # Usage
    p2d = Point2D(x=1, y=2)
    assert isinstance(p2d, Point2D)
    assert p2d.x == 1
    assert p2d.y == 2

    p3d = Point3D(x=3, y=4, z=5)
    assert isinstance(p3d, Point3D)
    assert p3d.x == 3
    assert p3d.y == 4
    assert p3d.z == 5

    gps = GPSCoordinate(latitude=40.7128, longitude=-74.0060)
    assert isinstance(gps, GPSCoordinate)
    assert gps.latitude == 40.7128
    assert gps.longitude == -74.0060
