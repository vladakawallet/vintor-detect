from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, WebSocketDisconnect, File, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor
from redis.asyncio import Redis

from core import DetectorCore
import logging
from typing import List
import asyncio




logger = logging.getLogger("detector-api")

def setup_logger(logger: logging.Logger):
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] %(funcName)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global r, core, executor
    setup_logger(logger)

    logger.info("Starting dependencies...")
    
    r = Redis(host="127.0.0.1", port=6379, decode_responses=True)
    core = DetectorCore()
    core.setup()
    executor = ThreadPoolExecutor(max_workers=3)

    yield
    if executor: executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():

    try:

        await r.ping()

        future = executor.submit(lambda: "ok")
        future.result(timeout=5)

        return JSONResponse(status_code=200, content={
            "status": "ok"
        })
    
    except Exception as e:
        logger.exception(f"Error on health check: {e}", exc_info=True)
        return JSONResponse(status_code=200, content={
            "status": "error",
            "message": str(e)
        })
    
@app.post("/images/{mode}/{id}")
async def receive_images(mode: str, id: str, files: List[UploadFile] = File(...)):

    try:
        if files and id:
            pipe = r.pipeline()
            for f in files:
                content = await f.read()
                pipe.rpush(f"car:{id}:images", content)
            pipe.set(f"car:{id}:status", "received")
            await pipe.execute()

            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(executor, core.process_car, id, mode)

            return JSONResponse(status_code=200, content={
                "status": "ok" if not future.cancelled() else "bad",
                "id": id})
        
    except Exception as e:
        logger.exception(f"Error while receiving the car {id}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    

@app.websocket("/ws/{id}")
async def websocket(websocket: WebSocket, id: str):
    await websocket.accept()
    try:
        while True:
            status = await r.get(f"car:{id}:status")
            
            if status in ("process", "received", "wait"): 
                await websocket.send_json({"status": status})

            elif status == "next": 
                await websocket.send_json({"status": status})
                await r.set(f"car:{id}:status", "wait", ex=600)

            elif status == "done": 
                res = await r.get(f"car:{id}:res")
                await websocket.send_json({
                    "status": "done", 
                    "result": res
                })
                break
            
            elif status == "error":
                await websocket.send_json({"status": status})
                break
            await asyncio.sleep(0.5)

    except WebSocketDisconnect as we:
        logger.warning(f"Car {id} disconnected with code {we.code} due to {we.reason}")
    except Exception as e:
        logger.exception(f"Websocket error on car {id}: {str(e)}")
        await websocket.send_json({"status": "error"})
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()