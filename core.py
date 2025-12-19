from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor

from fast_plate_ocr import LicensePlateRecognizer
from paddleocr import PaddleOCR, TextRecognition
import numpy as np
import torch
import cv2

from contextlib import asynccontextmanager
from redis.asyncio import Redis
import logging
import redis

from typing import List
import traceback
import asyncio
import os
import re
import time
import pathlib
import sys


BASE_DIR = pathlib.Path(__file__).parent
YOLO_DIR = BASE_DIR / "yolov5"
sys.path.append(str(YOLO_DIR))

from yolov5.utils.augmentations import letterbox

logger = logging.getLogger("vintor-detect")


default_plate_pattern = r"^[A-Z]{2}\d{4}[A-Z]{2}$"
temp_plate_pattern = r"^\d{2}[A-Z]{2}\d{4}$"
plate_pattern = re.compile(f"{default_plate_pattern}|{temp_plate_pattern}")

def setup_logger(logger: logging.Logger):
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] %(funcName)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath

class DetectorCore:
    def __init__(self):
        self._plate_model = None
        self._vin_model = None
        self._ocr = None
        self._onnx = None
        self._r = None

        self._pattern = plate_pattern
        setup_logger(logger)


    def setup(self):
        logger.info("Setting up models...")

        try:

            self._r = redis.Redis(host="127.0.0.1", port=6379)
            self._r.ping()

            self._plate_model = torch.hub.load(
                "yolov5", "custom", path=os.path.join(BASE_DIR, "models", "plate.pt"),
                source="local", device="cpu", force_reload=True
            )

            # self._vin_model = PaddleOCR(text_recognition_model_name="PP-OCRv5_mobile_rec", device="cpu")

            self._onnx = LicensePlateRecognizer('european-plates-mobile-vit-v2-model', device='cpu')
            logger.debug("Setup finished.")

            time.sleep(10)
        except Exception as e:
            logger.exception(f"Error during setup: {e}", exc_info=True)

    def get_images(self, id: str):
        image_bytes = self._r.lrange(f"car:{id}:images", 0, -1)
        images = []
        for b in image_bytes:
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        return images

    def detect_plate(self, images):
        crops = self._plate_model(images).crop(save=False)
        if not crops: return []

        res = []
        for c in crops:
            if c['conf'] > 0.7:
                img = np.ascontiguousarray(c['im'])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res.append((c['conf'], gray))
        return sorted(res, key=lambda x: x[0], reverse=True)[:3]

    def recognize_plate(self, crops):
        plates, confs = self._onnx.run([c[1] for c in crops], return_confidence=True)
        if len(plates) == 0 or len(confs) == 0: return

        res = []
        for p, c in zip(plates, confs):
            avg = round(sum(c) / len(c), 3)
            if avg >= 0.73:
                p = p.replace("_", "").strip()
                if self._pattern.match(p):
                    res.append((avg, p))

        if res: return max(res, key=lambda x: x[0])[1]

    def detect_vin(self, images):
        crops = self._plate_model(images).crop(save=False)
        if not crops: return []

        res = []
        for c in crops:
            if c['conf'] > 0.7:
                img = np.ascontiguousarray(c['im'])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res.append((c['conf'], gray))
        return sorted(res, key=lambda x: x[0], reverse=True)[:3]
    
    def recognize_vin(self, crops):
        res = self._vin_model.predict([c[1] for c in crops])
        for i in res:
            print(i)

    def del_car(self, id: str, status: str):
        pipe = self._r.pipeline()
        pipe.delete(f"car:{id}:images")
        pipe.set(f"car:{id}:status", status, ex=600)
        pipe.execute()

    def save_car(self, id: str, res: str):
        pipe = self._r.pipeline()
        pipe.delete(f"car:{id}:images")
        pipe.set(f"car:{id}:status", "done", ex=600)
        pipe.set(f"car:{id}:res", res, ex=600)
        pipe.execute()


    def process_car(self, id: str, mode: str, images = None):

        start = time.time()
        
        try:

            self._r.set(f"car:{id}:status", "process")
            
            if not images: images = self.get_images(id)
            if not images: 
                self.del_car(id, "next")
                return
            
            if mode == "plate":
                crops = self.detect_plate(images)
                if not crops:
                    self.del_car(id, "next")
                    return
                
                plate = self.recognize_plate(crops)
                if plate: 
                    self.save_car(id, plate)
                    return plate
                else:
                    self.del_car(id, "next")

        except Exception as e:
            self.del_car(id, "next")
            logger.exception(f"Error processing car {id}: {e}", exc_info=True)
        finally:
            logger.debug(f"[TIME] Processing time for {id}: {round(time.time() - start, 3)}s")

if __name__ == "__main__":
    if plate_pattern.match("17AX7912"): print("loggg")
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
    # api = DetectorCore()
    # api.setup()