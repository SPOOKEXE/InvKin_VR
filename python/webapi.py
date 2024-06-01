
from typing import Any, Callable, Union
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS,FACEMESH_FACE_OVAL,FACEMESH_IRISES,FACEMESH_LEFT_EYE,FACEMESH_LEFT_EYEBROW,FACEMESH_LEFT_IRIS,FACEMESH_LIPS,FACEMESH_NOSE,FACEMESH_RIGHT_EYE,FACEMESH_RIGHT_EYEBROW,FACEMESH_RIGHT_IRIS,FACEMESH_TESSELATION
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS,HAND_INDEX_FINGER_CONNECTIONS,HAND_MIDDLE_FINGER_CONNECTIONS,HAND_PALM_CONNECTIONS,HAND_PINKY_FINGER_CONNECTIONS,HAND_RING_FINGER_CONNECTIONS,HAND_THUMB_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

import time
import zlib
import json

ZLIB_COMPRESSION : bool = True

API_KEY : str = None
POSE_DATA : dict = None

def set_api_key( value : str ) -> None:
	global API_KEY
	API_KEY = value

def set_pose_data( data : dict ) -> None:
	global POSE_DATA
	POSE_DATA = data

async def validate_api_key( key : Union[str, None] = Security(APIKeyHeader(name="X-API-KEY", auto_error=False)) ) -> None:
	if API_KEY is not None and key != API_KEY:
		raise HTTPException(status_code=401, detail="Unauthorized")
	return None

tracking_api = FastAPI( title="Tracking API", description="Get any active tracker information.", version="1.0.0" )

@tracking_api.middleware("http")
async def process_time_adder( request : Request, call_next : Callable ) -> Any:
	start_time = time.time()
	response = await call_next(request)
	process_time = time.time() - start_time
	response.headers["X-Process-Time"] = str(process_time)
	return response

@tracking_api.get('/')
async def root() -> str:
	return "OK"

@tracking_api.get('/values', description="Get the body tracking values.", dependencies=[Depends(validate_api_key)])
async def get_values() -> dict:
	data = POSE_DATA
	if ZLIB_COMPRESSION is True:
		data = zlib.compress( json.dumps(data).encode('utf-8') ).hex()
	return { "data" : data }

@tracking_api.get('/contours', description="Get the body tracking contour values.", dependencies=[Depends(validate_api_key)])
async def get_contours() -> dict:
	def convert(v : list[tuple]) -> list[list]:
		return [list(item) for item in v]
	data = {
		"pose" : {
			"POSE_CONNECTIONS" : convert(POSE_CONNECTIONS)
		},
		"hand" : {
			"HAND_CONNECTIONS" : convert(HAND_CONNECTIONS),
			"HAND_INDEX_FINGER_CONNECTIONS" : convert(HAND_INDEX_FINGER_CONNECTIONS),
			"HAND_MIDDLE_FINGER_CONNECTIONS" : convert(HAND_MIDDLE_FINGER_CONNECTIONS),
			"HAND_PALM_CONNECTIONS" : convert(HAND_PALM_CONNECTIONS),
			"HAND_PINKY_FINGER_CONNECTIONS" : convert(HAND_PINKY_FINGER_CONNECTIONS),
			"HAND_RING_FINGER_CONNECTIONS" : convert(HAND_RING_FINGER_CONNECTIONS),
			"HAND_THUMB_CONNECTIONS" : convert(HAND_THUMB_CONNECTIONS),
		},
		"face_mesh" : {
			"FACEMESH_CONTOURS" : convert(FACEMESH_CONTOURS),
			"FACEMESH_FACE_OVAL" : convert(FACEMESH_FACE_OVAL),
			"FACEMESH_IRISES" : convert(FACEMESH_IRISES),
			"FACEMESH_LEFT_EYE" : convert(FACEMESH_LEFT_EYE),
			"FACEMESH_LEFT_EYEBROW" : convert(FACEMESH_LEFT_EYEBROW),
			"FACEMESH_LEFT_IRIS" : convert(FACEMESH_LEFT_IRIS),
			"FACEMESH_LIPS" : convert(FACEMESH_LIPS),
			"FACEMESH_NOSE" : convert(FACEMESH_NOSE),
			"FACEMESH_RIGHT_EYE" : convert(FACEMESH_RIGHT_EYE),
			"FACEMESH_RIGHT_EYEBROW" : convert(FACEMESH_RIGHT_EYEBROW),
			"FACEMESH_RIGHT_IRIS" : convert(FACEMESH_RIGHT_IRIS),
			"FACEMESH_TESSELATION" : convert(FACEMESH_TESSELATION),
		},
	}
	if ZLIB_COMPRESSION is True:
		data = zlib.compress( json.dumps(data).encode('utf-8') ).hex()
	return { "data" : data }
