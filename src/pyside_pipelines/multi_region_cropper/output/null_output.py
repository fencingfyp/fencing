from .region_output import RegionOutput


class NullOutput(RegionOutput):
    def process(self, frame, quad_np, frame_id):
        pass

    def close(self):
        pass

    def delete(self):
        pass
