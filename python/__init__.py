
from tracker import HandTracker, PoseTracker, FaceMeshTracker
from tracker import FaceMeshResults, HandTrackerResults, PoseTrackerResults
from tracker import preprocess_image, unprocess_image

import cv2

if __name__ == "__main__":

	handTracker = HandTracker()
	poseTracker = PoseTracker()
	faceMeshTracker = FaceMeshTracker()

	videoCapture = cv2.VideoCapture(0)
	while videoCapture.isOpened():
		success, image = videoCapture.read()
		if not success:
			continue

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = preprocess_image(image)

		handResults = handTracker.parse_image(image)
		poseResults = poseTracker.parse_image(image)
		faceMeshResults = faceMeshTracker.parse_image(image)

		image = unprocess_image(image)

		handTracker.draw_landmarks( image, handResults.landmarks )
		poseTracker.draw_landmarks( image, poseResults.landmarks )
		faceMeshTracker.draw_landmarks( image, faceMeshResults.landmarks )

		cv2.imshow('Image Feed', cv2.flip(image, 1))

		if cv2.waitKey(5) & 0xFF == 27:
			break
	videoCapture.release()
