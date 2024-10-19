import mediapipe as mp

class detection:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.85)
        self.mp_drawing = mp.solutions.drawing_utils

    def checkPoseEstimation(self, frame):
        results = self.pose.process(frame)
        if self.results.pose_landmarks:
            