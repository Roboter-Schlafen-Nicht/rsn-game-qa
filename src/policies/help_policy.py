# yolo_help_policy.py
import torch
from ultralytics import YOLO

CLASS_BTN_HELP = 2


class HelpOnlyPolicy:
    """Thin wrapper around YOLO to get the best BTN_HELP detection from
    a frame."""

    def __init__(self, model_path: str, device: str = "cpu", conf_thres: float = 0.4):
        """
        model_path: path to YOLO weights (e.g.
                    runs/detect/train9/weights/best.pt)
        device: 'cpu', 'cuda:0', 'xpu', etc.
        conf_thres: minimum confidence to consider a HELP detection.
        """
        self.device = torch.device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_thres = conf_thres
        print("[HELP POLICY] names:", self.model.names)

    def infer(self, frame_np):
        """
        Run YOLO on a single frame and return the best HELP detection.

        frame_np: HWC numpy array, BGR uint8
                  (from LastWarController.screencapnp).

        Returns:
            None if no suitable HELP, else dict:
            {
                "cx": int,   # center x in pixels
                "cy": int,   # center y in pixels
                "conf": float,
                "bbox": (x1, y1, x2, y2)
            }
        """
        # Ultralytics supports numpy HWC images directly.[web:266][web:269]
        results = self.model.predict(
            source=frame_np,
            device=self.device,
            conf=self.conf_thres,
            imgsz=640,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return None

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        # (N, 4) [x1, y1, x2, y2][web:272][web:278]
        xyxy = boxes.xyxy.cpu().numpy()
        best = None
        best_conf = -1.0
        for cid, conf, (x1, y1, x2, y2) in zip(cls_ids, confs, xyxy):
            if cid != CLASS_BTN_HELP:
                continue
            if conf < self.conf_thres:
                continue

            if best is None or conf > best_conf:
                best = (float(x1), float(y1), float(x2), float(y2))
                best_conf = float(conf)

        if best is None:
            return None

        x1, y1, x2, y2 = best
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)

        return {
            "conf": best_conf,
            "bbox": (x1, y1, x2, y2),
            "cx": cx,
            "cy": cy,
        }
