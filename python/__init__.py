
from fastapi import FastAPI
from PIL import Image, ImageTk

from tracker import HandTracker, PoseTracker, FaceMeshTracker
from tracker import FaceMeshResults, HandTrackerResults, PoseTrackerResults
from tracker import FaceMeshTrackerSettings, HandTrackerSettings, PoseTrackerSettings
from tracker import preprocess_image, unprocess_image
from webapi import set_api_key, set_pose_data, tracking_api

import cv2
import uvicorn
import asyncio
import numpy as np
import tkinter as tk

async def host_fastapp( app : FastAPI, host : str, port : int ) -> None:
	print(f"Hosting App: {app.title}")
	await uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level='debug')).serve()

async def parse_detection(
	image : np.ndarray,
	hand_tracker : HandTracker,
	pose_tracker : PoseTracker,
	face_mesh_tracker : FaceMeshTracker
) -> None:
	image = preprocess_image(image)

	handResults : HandTrackerResults = hand_tracker.parse_image(image)
	poseResults : PoseTrackerResults = pose_tracker.parse_image(image)
	faceMeshResults : FaceMeshResults = face_mesh_tracker.parse_image(image)

	set_pose_data({
		"settings" : {
			"hand" : hand_tracker.settings.model_dump(),
			"pose" : pose_tracker.settings.model_dump(),
			"face_mesh" : face_mesh_tracker.settings.model_dump(),
		},
		"results" : {
			"hand" : handResults.model_dump(),
			"pose" : poseResults.model_dump(),
			"face_mesh" : faceMeshResults.model_dump(),
		}
	})

	image = unprocess_image(image)

	hand_tracker.draw_landmarks( image, handResults.landmarks )
	pose_tracker.draw_landmarks( image, poseResults.landmarks )
	face_mesh_tracker.draw_landmarks( image, faceMeshResults.landmarks )

	return image

class AsyncTkCanvas:

	window_closed : bool
	canvas_size : tuple[int, int]

	def __init__( self, width : int, height : int ) -> None:
		self.window_closed = False
		self.canvas_size = (width, height)
		self.root = tk.Tk()
		self.canvas = tk.Canvas(self.root, width=width, height=height)
		self.canvas.pack()

		def on_close():
			self.window_closed = True
		self.root.protocol("WM_DELETE_WINDOW", on_close)

	async def load_image( self, image : Image.Image ) -> None:
		image.thumbnail(self.canvas_size)
		photo = ImageTk.PhotoImage(image)
		self.canvas.delete("all")
		self.canvas.create_image( 0, 0, anchor=tk.NW, image=photo )
		self.canvas.image = photo

	async def load_image_np( self, image : np.ndarray ) -> Image.Image:
		return await self.load_image( Image.fromarray(image) )

	async def update( self ) -> None:
		self.root.update_idletasks()
		self.root.update()

async def tracker_main() -> None:
	videoCapture = cv2.VideoCapture(1)
	handTracker = HandTracker( HandTrackerSettings() )
	poseTracker = PoseTracker( PoseTrackerSettings() )
	faceMeshTracker = FaceMeshTracker( FaceMeshTrackerSettings() )

	display_canvas = AsyncTkCanvas(800, 800)
	display_canvas.root.title('Image Viewer')

	while videoCapture.isOpened() is True and display_canvas.window_closed is False:
		if display_canvas.window_closed is True:
			break
		success, image = videoCapture.read()
		if success is True:
			image = await parse_detection( image, handTracker, poseTracker, faceMeshTracker )
			await display_canvas.load_image_np(image)
		await display_canvas.update()
		await asyncio.sleep(0.01)

	display_canvas.root.destroy()
	videoCapture.release()

async def fastapi_main( host : str = '127.0.0.1', port : int = 5100, api_key : str = None ) -> None:
	set_api_key(api_key)
	await host_fastapp(tracking_api, host, port)

async def main(host : str = '127.0.0.1', port : int = 5100, api_key : str = None) -> None:
	print(f'Setting API_Key to "{api_key}"')
	t1 = asyncio.create_task(tracker_main())
	t2 = asyncio.create_task(fastapi_main(host=host, port=port, api_key=api_key))
	await asyncio.gather(t1, t2)

if __name__ == "__main__":
	asyncio.run(main())
