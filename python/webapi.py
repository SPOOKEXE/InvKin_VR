
from typing import Any, Callable, Union
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

import time

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
	return { "data" : POSE_DATA }
