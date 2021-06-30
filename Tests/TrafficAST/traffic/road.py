from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from traffic.constants import *

class RoadSegment:
    def __init__(self,vertices):
        self.vertices = vertices
        self.polygon = Polygon(self.vertices)

    def is_in(self,car):
        point = Point(car.position[0],car.position[1])
        return self.polygon.contains(point)

    def setup_render(self, viewer):
        from gym.envs.classic_control import rendering
        road_poly = self.vertices
        self.geom = rendering.make_polygon(road_poly)
        self.xform = rendering.Transform()
        self.geom.add_attr(self.xform)
        self.geom.set_color(*ROAD_COLOR)
        viewer.add_geom(self.geom)

    def update_render(self, camera_center):
        self.xform.set_translation(*(- camera_center))

class Road:
    def __init__(self,segments):
        self.segments=segments

    def is_in(self,car):
        for segment in self.segments:
            if segment.is_in(car):
                return True
        return False

    def setup_render(self, viewer):
        for segment in self.segments:
            segment.setup_render(viewer)

    def update_render(self, camera_center):
        for segment in self.segments:
            segment.update_render(camera_center)

