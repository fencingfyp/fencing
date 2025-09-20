import cv2
import numpy as np

MIN_INLIERS = 15  # tune as needed


class PlanarTarget:
    def __init__(self, frame, initial_positions, orb=None, bf=None):
        # reuse global ORB/Matcher if passed, else create new
        self.orb = orb or cv2.ORB_create(1000)
        self.bf = bf or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(gray, None)

        self.initial_positions = initial_positions
        self.last_quad = initial_positions

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) == 0:
            return self.last_quad

        matches = self.bf.match(self.des_ref, des_frame)
        if len(matches) < MIN_INLIERS:
            return self.last_quad

        # take only best N matches to save time
        matches = sorted(matches, key=lambda x: x.distance)[:200]

        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            inliers = mask.sum()
            if inliers > MIN_INLIERS:
                self.last_quad = cv2.perspectiveTransform(self.initial_positions, H)

        return self.last_quad


class TargetsTracker:
    def __init__(self):
        # share one ORB + BFMatcher across all targets
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.targets = {}

    def add_target(self, name, frame, initial_positions):
        target = PlanarTarget(frame, initial_positions, self.orb, self.bf)
        self.targets[name] = target

    def update_all(self, frame):
        outputs = {}
        for name, target in self.targets.items():
            outputs[name] = target.update(frame)
        return outputs
