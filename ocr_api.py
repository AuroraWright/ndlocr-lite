import sys
import os
import time
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

from .deim import DEIM
from .parseq import PARSEQ
from yaml import safe_load
from .reading_order.xy_cut.eval import eval_xml
from .ndl_parser import convert_to_xml_string3

logger = logging.getLogger(__name__)
base_dir = Path(__file__).resolve().parent

class RecogLine:
    def __init__(self, npimg: np.ndarray, idx: int, pred_char_cnt: int, pred_str: str = ""):
        self.npimg = npimg
        self.idx   = idx
        self.pred_char_cnt = pred_char_cnt
        self.pred_str = pred_str

    def __lt__(self, other):  
        return self.idx < other.idx

def process_cascade(alllineobj, recognizer30, recognizer50, recognizer100, is_cascade=True):
    targetdflist30, targetdflist50, targetdflist100 = [], [], []
    for lineobj in alllineobj:
        if lineobj.pred_char_cnt == 3 and is_cascade:
            targetdflist30.append(lineobj)
        elif lineobj.pred_char_cnt == 2 and is_cascade:
            targetdflist50.append(lineobj)
        else:
            targetdflist100.append(lineobj)
            
    targetdflistall = []
    with ThreadPoolExecutor(thread_name_prefix="thread") as executor:
        resultlines30, resultlines50, resultlines100 = [], [], []
        
        if len(targetdflist30) > 0:
            resultlines30 = list(executor.map(recognizer30.read, [t.npimg for t in targetdflist30]))
        for i in range(len(targetdflist30)):
            pred_str = resultlines30[i]
            lineobj = targetdflist30[i]
            if len(pred_str) >= 25:
                targetdflist50.append(lineobj)
            else:
                lineobj.pred_str = pred_str
                targetdflistall.append(lineobj)
                
        if len(targetdflist50) > 0:
            resultlines50 = list(executor.map(recognizer50.read, [t.npimg for t in targetdflist50]))
        for i in range(len(targetdflist50)):
            pred_str = resultlines50[i]
            lineobj = targetdflist50[i]
            if len(pred_str) >= 45:
                targetdflist100.append(lineobj)
            else:
                lineobj.pred_str = pred_str
                targetdflistall.append(lineobj)
                
        if len(targetdflist100) > 0:
            resultlines100 = list(executor.map(recognizer100.read, [t.npimg for t in targetdflist100]))
        for i in range(len(targetdflist100)):
            pred_str = resultlines100[i]
            lineobj = targetdflist100[i]
            lineobj.pred_str = pred_str
            targetdflistall.append(lineobj)                    
            
        targetdflistall = sorted(targetdflistall)
        resultlinesall = [t.pred_str for t in targetdflistall]
        
    return resultlinesall


class OCRPipeline:
    def __init__(self, 
                 det_weights: str = str(Path.home() / ".config" / "ndlocr_lite" / "deim-s-1024x1024.onnx"), 
                 det_classes: str = str(base_dir / "config" / "ndl.yaml"), 
                 rec_weights30: str = str(Path.home() / ".config" / "ndlocr_lite" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"), 
                 rec_weights50: str = str(Path.home() / ".config" / "ndlocr_lite" / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx"), 
                 rec_weights100: str = str(Path.home() / ".config" / "ndlocr_lite" / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx"), 
                 rec_classes: str = str(base_dir / "config" / "NDLmoji.yaml"), 
                 device: str = "cpu",
                 det_score_threshold: float = 0.2, 
                 det_conf_threshold: float = 0.25, 
                 det_iou_threshold: float = 0.2):
        
        # 1. Initialize Detector
        logger.debug("Loading Detector...")
        self.detector = DEIM(model_path=det_weights,
                             class_mapping_path=det_classes,
                             score_threshold=det_score_threshold,
                             conf_threshold=det_conf_threshold,
                             iou_threshold=det_iou_threshold,
                             device=device)
        self.classeslist = list(self.detector.classes.values())
        
        # 2. Extract charset
        logger.debug("Loading Recognizers...")
        with open(rec_classes, encoding="utf-8") as f:
            charobj = safe_load(f)
        charlist = list(charobj["model"]["charset_train"])
        
        # 3. Initialize Recognizers
        self.recognizer30 = PARSEQ(model_path=rec_weights30, charlist=charlist, device=device)
        self.recognizer50 = PARSEQ(model_path=rec_weights50, charlist=charlist, device=device)
        self.recognizer100 = PARSEQ(model_path=rec_weights100, charlist=charlist, device=device)
        logger.debug("Models loaded successfully.")

    def process_image(self, pil_image: Image.Image) -> dict:
        """
        Takes a PIL image directly from memory, runs the OCR pipeline, 
        and returns a structured JSON-like dictionary containing the results.
        """
        start = time.time()
        
        # Convert PIL to Numpy
        img = np.array(pil_image.convert('RGB'))
        img_h, img_w = img.shape[:2]
        
        # 1. Run Detector
        detections = self.detector.detect(img)
        
        resultobj = [dict(), dict()]
        resultobj[0][0] = list()
        for i in range(17):
            resultobj[1][i] = []
            
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            conf = det["confidence"]
            if det["class_index"] == 0:
                resultobj[0][0].append([xmin, ymin, xmax, ymax])
            resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf])
            
        # 2. Calculate Reading Order via XML (In-Memory)
        # We pass a dummy name since we aren't saving it
        xmlstr = convert_to_xml_string3(img_w, img_h, "memory_image", self.classeslist, resultobj)
        xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
        root = ET.fromstring(xmlstr)
        eval_xml(root, logger=None)
        
        # 3. Extract Lines
        alllineobj = []
        tatelinecnt = 0
        alllinecnt = 0

        for idx, lineobj in enumerate(root.findall(".//LINE")):
            xmin, ymin = int(lineobj.get("X")), int(lineobj.get("Y"))
            line_w, line_h = int(lineobj.get("WIDTH")), int(lineobj.get("HEIGHT"))
            
            pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT", 100.0))
            if line_h > line_w: tatelinecnt += 1
            alllinecnt += 1
            
            # Crop image slice
            lineimg = img[ymin:ymin+line_h, xmin:xmin+line_w, :]
            alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

        # Handle fallback: Detections exist but no LINE elements generated
        if len(alllineobj) == 0 and len(detections) > 0:
            page = root.find("PAGE")
            for idx, det in enumerate(detections):
                xmin, ymin, xmax, ymax = det["box"]
                line_w, line_h = int(xmax - xmin), int(ymax - ymin)
                
                if line_w > 0 and line_h > 0:
                    line_elem = ET.SubElement(page, "LINE")
                    line_elem.set("TYPE", "本文")
                    line_elem.set("X", str(int(xmin)))
                    line_elem.set("Y", str(int(ymin)))
                    line_elem.set("WIDTH", str(line_w))
                    line_elem.set("HEIGHT", str(line_h))
                    line_elem.set("CONF", f"{det['confidence']:0.3f}")
                    
                    pred_char_cnt = det.get("pred_char_count", 100.0)
                    line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
                    
                    if line_h > line_w: tatelinecnt += 1
                    alllinecnt += 1
                    
                    lineimg = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
                    alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

        # 4. Cascade Recognition
        resultlinesall = process_cascade(
            alllineobj, self.recognizer30, self.recognizer50, self.recognizer100, is_cascade=True
        )

        # 5. Format Output
        contents = []
        
        # We need a stable mapping from the XML line object to its numerical index 
        # so we fetch the correct recognized string from resultlinesall.
        all_lines_flat = root.findall(".//LINE")
        line_to_idx = {line: i for i, line in enumerate(all_lines_flat)}
        processed_lines = set()

        # Group lines by Paragraph (TEXTBLOCK)
        for textblock in root.findall(".//TEXTBLOCK"):
            tb_json_array = []
            for lineobj in textblock.findall(".//LINE"):
                idx = line_to_idx[lineobj]
                processed_lines.add(idx)
                
                xmin, ymin = int(lineobj.get("X")), int(lineobj.get("Y"))
                line_w, line_h = int(lineobj.get("WIDTH")), int(lineobj.get("HEIGHT"))
                conf = float(lineobj.get("CONF", 0.0))
                
                jsonobj = {
                    "boundingBox": [[xmin, ymin], [xmin+line_w, ymin], [xmin+line_w, ymin+line_h], [xmin, ymin+line_h]],
                    "id": idx,
                    "text": resultlinesall[idx],
                    "confidence": conf
                }
                tb_json_array.append(jsonobj)
                
            if tb_json_array:
                contents.append(tb_json_array)

        # Capture any lines that were NOT wrapped in a TEXTBLOCK 
        # (e.g., from the fallback block in Step 3 where lines are attached to PAGE)
        for lineobj in all_lines_flat:
            idx = line_to_idx[lineobj]
            if idx not in processed_lines:
                xmin, ymin = int(lineobj.get("X")), int(lineobj.get("Y"))
                line_w, line_h = int(lineobj.get("WIDTH")), int(lineobj.get("HEIGHT"))
                conf = float(lineobj.get("CONF", 0.0))
                
                jsonobj = {
                    "boundingBox": [[xmin, ymin], [xmin+line_w, ymin], [xmin+line_w, ymin+line_h], [xmin, ymin+line_h]],
                    "id": idx,
                    "text": resultlinesall[idx],
                    "confidence": conf
                }
                contents.append([jsonobj])

        processing_time = time.time() - start

        # Construct final dict
        result_payload = {
            "contents": contents,
            "processing_time_sec": processing_time
        }

        return result_payload
