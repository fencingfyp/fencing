import cv2
import csv
from run_pose_estimation_1 import CSV_COLS, NUM_KEYPOINTS

"""
This class manages loading and caching frame data from a CSV file.
It preloads a certain number of frames ahead to allow "looking into the future",
to 1: check if a fencer id reappears in the future frames, and 2: check the scoreboard
in future frames if a score event is detected.
"""

class FrameInfoManager:
  def __init__(self, csv_path: str, fps: int, num_ms_ahead: int = 10000):
    self.num_ms_ahead = num_ms_ahead
    self.frame_records = {}
    self.current_frame_index = 0
    self.loaded_up_to_frame = -1
    self.frame_reader = self.read_csv_by_frame(csv_path)
    self.fps = fps
    self.num_frames_ahead = max(1, int(self.fps * (self.num_ms_ahead / 1000)))
    print(f"FrameInfoManager initialized to load {self.num_frames_ahead} frames ahead at {self.fps} fps")
  @staticmethod
  def read_csv_by_frame(path):
    with open(path, newline="") as f:
      reader = csv.reader(f)
      header = next(reader)

      # Ensure correct number of columns
      if len(header) != CSV_COLS:
        raise ValueError(f"CSV has {len(header)} columns, expected {CSV_COLS}")

      current_frame = None
      batch = []

      for row in reader:
        frame_id = int(row[0])  # frame_id

        if current_frame is None:
          current_frame = frame_id

        if frame_id != current_frame:
          yield current_frame, batch
          batch = []
          current_frame = frame_id

        # Convert row to dict
        id = int(row[1])
        conf = float(row[2])
        box = list(map(float, row[3:7]))
        keypoints = []
        kp_vals = row[7:]

        for i in range(NUM_KEYPOINTS):
          x = float(kp_vals[i*3 + 0])
          y = float(kp_vals[i*3 + 1])
          v = float(kp_vals[i*3 + 2])
          keypoints.append((x, y, v))

        batch.append({
          "id": id,
          "confidence": conf,
          "box": box,  # [x1, y1, x2, y2]
          "keypoints": keypoints
        })

      if batch:
        yield current_frame, batch

  def preload_detections_at_frame(self, frame_index: int):
    # load frames up to frame_index + num_frames_ahead
    target_index = frame_index + self.num_frames_ahead
    while self.loaded_up_to_frame < target_index:
      frame_num, batch = next(self.frame_reader, (None, None))
      if frame_num is None:
        break
      self.loaded_up_to_frame += 1
      self.frame_records[frame_num] = {det["id"]: det for det in batch}
    return self.frame_records.get(frame_index)
  
  # we only load forward and preload when the current frame is less than or equal to the requested frame
  def get_detections(self, frame_index: int) -> dict[int, dict] | None:
    # delete previous frames to save memory, assuming only the previous frame hasn't been deleted
    self.frame_records.pop(frame_index - 1, None)
    if frame_index < self.current_frame_index:
      raise ValueError("Can only get frames in increasing order")
    if frame_index == self.current_frame_index:
      self.current_frame_index += 1
      return self.preload_detections_at_frame(frame_index)
    return self.search_frame(frame_index)
  
  # For "searching in the future" functionality, we only allow searching for already loaded frames
  def search_frame(self, frame_index: int) -> dict[int, dict] | None:
    if frame_index in self.frame_records:
      return self.frame_records[frame_index]
    raise ValueError("Can only search for already loaded frames")
  
  def appears_in_future_detections(self, current_frame_index: int, fencer_id: int) -> bool:
    # check if fencer_id appears in any of the preloaded future frames
    for fi in range(current_frame_index + 1, self.loaded_up_to_frame + 1):
      frame_dets = self.frame_records.get(fi)
      if frame_dets and fencer_id in frame_dets:
        print(f"Fencer {fencer_id} reappears in frame {fi}")
        return True
    return False
    

