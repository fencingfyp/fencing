import numpy as np


class RegionOutput:
    """
    Base class for any region output.
    """

    def process(
        self,
        frame: np.ndarray,
        quad_np: np.ndarray,
        frame_id: int,
    ):
        raise NotImplementedError

    def close(self):
        pass

    def delete(self):
        pass
