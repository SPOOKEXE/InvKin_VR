
from typing import Any, Mapping, NamedTuple, Tuple
from PIL import Image
from pydantic import BaseModel
from pyparsing import abstractmethod

from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.hands import Hands, HandLandmark
from mediapipe.python.solutions.pose import Pose, PoseLandmark
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS,FACEMESH_FACE_OVAL,FACEMESH_IRISES,FACEMESH_LEFT_EYE,FACEMESH_LEFT_EYEBROW,FACEMESH_LEFT_IRIS,FACEMESH_LIPS,FACEMESH_NOSE,FACEMESH_RIGHT_EYE,FACEMESH_RIGHT_EYEBROW,FACEMESH_RIGHT_IRIS,FACEMESH_TESSELATION
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS,HAND_INDEX_FINGER_CONNECTIONS,HAND_MIDDLE_FINGER_CONNECTIONS,HAND_PALM_CONNECTIONS,HAND_PINKY_FINGER_CONNECTIONS,HAND_RING_FINGER_CONNECTIONS,HAND_THUMB_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import WHITE_COLOR, RED_COLOR, _BGR_CHANNELS, _VISIBILITY_THRESHOLD, _PRESENCE_THRESHOLD, _normalized_to_pixel_coordinates
from mediapipe.python.solutions.drawing_styles import DrawingSpec, get_default_hand_connections_style, get_default_face_mesh_iris_connections_style, get_default_face_mesh_tesselation_style, get_default_face_mesh_contours_style, get_default_pose_landmarks_style

import cv2
import numpy as np

class NormalizedLandmark(BaseModel):
	x : float
	y : float
	z : float

class HandTrackerSettings(BaseModel):
	model_complexity : int = 0
	min_detection_confidence : float = 0.3
	min_tracking_confidence : float = 0.3

class PoseTrackerSettings(BaseModel):
	min_detection_confidence : float = 0.3
	min_tracking_confidence : float = 0.3

class FaceMeshTrackerSettings(BaseModel):
	max_num_faces : int = 1
	refine_landmarks : bool = True
	min_detection_confidence : float = 0.3
	min_tracking_confidence : float = 0.3

class Results(BaseModel):
	pass

class PoseTrackerResults(Results):
	landmarks : list[NormalizedLandmark]
	landmarks_3d : list[NormalizedLandmark]

class HandTrackerResults(Results):
	landmarks : list[list[NormalizedLandmark]]

class FaceMeshResults(Results):
	landmarks : list[NormalizedLandmark]

def preprocess_image( image : np.ndarray ) -> np.ndarray:
	# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	image.flags.writeable = False # slight performance boost
	return image

def unprocess_image( image : np.ndarray ) -> np.ndarray:
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = True # slight performance boost
	return image

def contour_connections( points : list, contours : frozenset[tuple] ) -> list[tuple[NormalizedLandmark, NormalizedLandmark]]:
	return [ (points[ pair[0] ], points[ pair[1] ]) for pair in contours ]

def _draw_landmarks(
	image,
	landmark_list : list[NormalizedLandmark],
	connections = None,
	landmark_drawing_spec = DrawingSpec(color=RED_COLOR),
	connection_drawing_spec = DrawingSpec(),
	is_drawing_landmarks : bool = True,
) -> None:
	if not landmark_list:
		return
	if image.shape[2] != _BGR_CHANNELS:
		raise ValueError('Input image must contain three channel bgr data.')
	image_rows, image_cols, _ = image.shape
	idx_to_coordinates = {}
	for idx, landmark in enumerate(landmark_list):
		landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
		if landmark_px:
			idx_to_coordinates[idx] = landmark_px
	if connections:
		num_landmarks = len(landmark_list)
		for connection in connections:
			start_idx = connection[0]
			end_idx = connection[1]
			if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
				raise ValueError(f'Landmark index is out of range. Invalid connection from landmark #{start_idx} to landmark #{end_idx}.')
			if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
				drawing_spec = connection_drawing_spec[connection] if isinstance(connection_drawing_spec, Mapping) else connection_drawing_spec
				cv2.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], drawing_spec.color, drawing_spec.thickness)
	if is_drawing_landmarks and landmark_drawing_spec:
		for idx, landmark_px in idx_to_coordinates.items():
			drawing_spec = landmark_drawing_spec[idx] if isinstance(landmark_drawing_spec, Mapping) else landmark_drawing_spec
			circle_border_radius = max(drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
			cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR, drawing_spec.thickness)
			cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)

class Tracker:

	def __init__( self ) -> None:
		pass

	@abstractmethod
	def parse_image( self, image : Image.Image ) -> dict:
		pass

	@abstractmethod
	def draw_landmarks( self, image : np.ndarray ) -> np.ndarray:
		pass

class FaceMeshTracker(Tracker):
	settings : FaceMeshTrackerSettings
	face_mesh : FaceMesh

	def __init__( self, settings : FaceMeshTrackerSettings ) -> None:
		self.settings = settings
		self.face_mesh = FaceMesh( **settings.model_dump() )

	def parse_image( self, image : np.ndarray ) -> FaceMeshResults:
		results = self.face_mesh.process( image )
		if results.multi_face_landmarks is not None:
			landmarks = [ NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for face_landmarks in results.multi_face_landmarks for landmark in face_landmarks.landmark ]
		else:
			landmarks = []
		return FaceMeshResults(landmarks=landmarks)

	def draw_landmarks( self, image : np.ndarray, face_landmarks : list[NormalizedLandmark] ) -> np.ndarray:
		_draw_landmarks(
			image=image,
			landmark_list=face_landmarks,
			connections=FACEMESH_TESSELATION,
			landmark_drawing_spec=None,
			connection_drawing_spec=get_default_face_mesh_tesselation_style()
		)
		_draw_landmarks(
			image=image,
			landmark_list=face_landmarks,
			connections=FACEMESH_CONTOURS,
			landmark_drawing_spec=None,
			connection_drawing_spec=get_default_face_mesh_contours_style()
		)
		_draw_landmarks(
			image=image,
			landmark_list=face_landmarks,
			connections=FACEMESH_IRISES,
			landmark_drawing_spec=None,
			connection_drawing_spec=get_default_face_mesh_iris_connections_style()
		)

class PoseTracker(Tracker):
	settings : PoseTrackerSettings
	pose : Pose

	def __init__( self, settings : PoseTrackerSettings ) -> None:
		self.settings = settings
		self.pose = Pose( **settings.model_dump() )

	def parse_image( self, image : np.ndarray ) -> PoseTrackerResults:
		results = self.pose.process( image ) # pose_landmarks, pose_world_landmarks
		if results.pose_landmarks is not None:
			landmarks = [NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in results.pose_landmarks.landmark]
		else:
			landmarks = []
		if results.pose_world_landmarks is not None:
			landmarks_3d = [NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in results.pose_world_landmarks.landmark]
		else:
			landmarks_3d = []
		return PoseTrackerResults(landmarks=landmarks, landmarks_3d=landmarks_3d)

	def draw_landmarks( self, image : np.ndarray, pose_landmarks : list[NormalizedLandmark] ) -> None:
		_draw_landmarks(
			image,
			landmark_list=pose_landmarks,
			connections=POSE_CONNECTIONS,
			landmark_drawing_spec=get_default_pose_landmarks_style(),
		)

class HandTracker(Tracker):
	settings : HandTrackerSettings
	hands : Hands

	def __init__( self, settings : HandTrackerSettings ) -> None:
		self.settings = settings
		self.hands = Hands( **settings.model_dump() )

	def parse_image( self, image : np.ndarray ) -> HandTrackerResults:
		results = self.hands.process( image )
		landmarks = []
		if results.multi_hand_landmarks is not None:
			for hand_landmarks in results.multi_hand_landmarks:
				hand_marks : list = []
				for landmark in hand_landmarks.landmark:
					hand_marks.append(NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z))
				landmarks.append(hand_marks)
		landmarks_3d = []
		if results.multi_hand_world_landmarks is not None:
			for hand_world_landmarks in results.multi_hand_world_landmarks:
				hand_marks : list = []
				for landmark in hand_world_landmarks.landmark:
					hand_marks.append(NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z))
				landmarks_3d.append(hand_marks)
		return HandTrackerResults(landmarks=landmarks, landmarks_3d=landmarks_3d)

	def draw_landmarks( self, image : np.ndarray, pose_landmarks : list[list[NormalizedLandmark]] ) -> None:
		for landmark_array in pose_landmarks:
			_draw_landmarks(
				image,
				landmark_list=landmark_array,
				connections=HAND_CONNECTIONS,
				#landmark_drawing_spec=get_default_hand_connections_style(),
				connection_drawing_spec=get_default_hand_connections_style(),
			)

def test_faceTracker( image : np.ndarray ) -> None:
	image = preprocess_image( image )
	faceTracker : FaceMeshTracker = FaceMeshTracker()
	fm_results : FaceMeshResults = faceTracker.parse_image(image)
	image = unprocess_image( image )
	faceTracker.draw_landmarks(image, fm_results.landmarks)
	Image.fromarray( image ).show()

def test_poseTracker( image : np.ndarray ) -> None:
	image = preprocess_image( image )
	poseTracker : PoseTracker = PoseTracker()
	pt_results : PoseTrackerResults = poseTracker.parse_image(image)
	image = unprocess_image( image )
	poseTracker.draw_landmarks(image, pt_results.landmarks)
	Image.fromarray( image ).show()

def test_handTracker( image : np.ndarray ) -> None:
	image = preprocess_image( image )
	handTracker : HandTracker = HandTracker()
	ht_results : HandTrackerResults = handTracker.parse_image(image)
	image = unprocess_image( image )
	handTracker.draw_landmarks(image, ht_results.landmarks)
	Image.fromarray( image ).show()
