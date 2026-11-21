"""Run from repo root after starting the API: python tests/check_api.py."""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


def run_checks(client):
    checked = 0

    def check(condition, description):
        nonlocal checked
        assert condition, description
        checked += 1

    response = client.get("/health")
    check(response.status_code == 200, "Health endpoint")
    check("trash" in response.json()["tasks"], "Trash model loaded")
    task = next(t for t in client.get("/tasks").json()["tasks"] if t["name"] == "trash")
    known = json.loads((ROOT / "data/trash/known_answers.json").read_text())
    check(len(known) == 12, "12 paired reference samples")
    tolerance = task.get("gradcam_tolerance", 1e-4)
    for name, reference in known.items():
        data = (ROOT / "data/trash/samples" / name).read_bytes()
        response = client.post("/tasks/trash/predict", files={"file": (name, data)})
        check(response.status_code == 200, f"Prediction {name}")
        result = response.json()
        p = [result["probabilities"][c] for c in task["classes"]]
        check(np.allclose(p, reference["probabilities"], atol=tolerance, rtol=0), f"Notebook probabilities {name}")
        check(np.isclose(sum(p), 1, atol=1e-5), f"Probability sum {name}")
        expected = result["top_class"] if max(p) >= task["not_sure_threshold"] else "not_sure"
        check(result["decision"] == expected, f"Threshold decision {name}")
        response = client.post("/tasks/trash/explain", files={"file": (name, data)})
        check(response.status_code == 200, f"Explanation {name}")
        explanation = response.json()
        check(np.allclose(explanation["cam"], reference["cam"], atol=tolerance, rtol=0), f"Notebook heatmap {name}")
        Image.open(io.BytesIO(base64.b64decode(explanation["overlay_png"]))).verify()
        check(True, f"Valid overlay PNG {name}")
    data = (ROOT / "data/trash/samples" / next(iter(known))).read_bytes()
    for route, files, expected in [
        ("/tasks/unknown/predict", {"file": ("a.jpg", data)}, 404),
        ("/tasks/trash/predict", {"file": ("a.jpg", b"")}, 400),
        ("/tasks/trash/predict", {"file": ("a.txt", b"not an image")}, 400),
        ("/tasks/trash/predict?threshold=2", {"file": ("a.jpg", data)}, 422),
        ("/tasks/trash/predict", None, 422),
    ]:
        check(client.post(route, files=files).status_code == expected, f"Invalid request {route}: {expected}")
    print(f"ALL PASS: {checked} checks. API predictions and Grad-CAM match the paired notebook export.")


if __name__ == "__main__":
    class LiveClient:
        def get(self, path):
            return requests.get(API + path, timeout=60)

        def post(self, path, **kwargs):
            return requests.post(API + path, timeout=60, **kwargs)

    run_checks(LiveClient())
