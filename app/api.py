import json
from pathlib import Path
from contextlib import asynccontextmanager
import base64
import io
import numpy as np
from PIL import Image
import keras
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
TASKS = {}


@asynccontextmanager
async def lifespan(app):
    folder = ROOT / "models/trash"
    meta = json.loads(
        (folder / "meta.json").read_text(encoding="utf-8")
    )
    model = keras.models.load_model(
        folder / meta["model_file"], compile=False
    )

    if len(meta["classes"]) != model.output_shape[-1]:
        raise ValueError("Metadata classes do not match model outputs.")

    TASKS["trash"] = {"model": model, "meta": meta}
    yield
    TASKS.clear()


app = FastAPI(
    title="Class 7 Waste Sorting",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok", "tasks": list(TASKS)}


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": name, **task["meta"]}
            for name, task in TASKS.items()
        ]
    }


@app.post("/tasks/{name}/predict")
async def predict(
    name: str,
    file: UploadFile = File(...),
    threshold: float | None = Query(None, ge=0, le=1)
):
    if name not in TASKS:
        raise HTTPException(404, "Unknown task")

    data = await file.read()

    if not data:
        raise HTTPException(400, "Empty image")

    task = TASKS[name]
    meta = task["meta"]

    try:
        image = tf.io.decode_image(
            data, channels=3, expand_animations=False
        )
        size = meta["image_size"]
        x = tf.image.resize(image, [size, size])[None]
    except (tf.errors.OpError, ValueError):
        raise HTTPException(400, "Invalid image")

    p = task["model"].predict(x, verbose=0)[0]
    t = meta["not_sure_threshold"] if threshold is None else threshold
    top = meta["classes"][int(p.argmax())]

    return {
        "task": name,
        "top_class": top,
        "probability": float(p.max()),
        "probabilities": {
            c: float(v) for c, v in zip(meta["classes"], p)
        },
        "decision": top if p.max() >= t else "not_sure",
        "threshold": t,
        "model": meta["model_name"]
    }
def gradcam(model, x):
    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, keras.Model)
    )
    head = model.get_layer("head")

    with tf.GradientTape() as tape:
        maps = backbone(
            model.get_layer("rescaling")(x),
            training=False
        )
        tape.watch(maps)

        scores = (
            tf.reduce_mean(maps, axis=(1, 2))
            @ head.kernel + head.bias
        )
        top = int(tf.argmax(scores[0]))
        selected = scores[:, top]

    grads = tape.gradient(selected, maps)
    weights = tf.reduce_mean(grads, axis=(1, 2))

    cam = tf.nn.relu(
        tf.reduce_sum(
            maps * weights[:, None, None, :],
            axis=-1
        )
    )[0].numpy()

    return (
        tf.nn.softmax(scores)[0].numpy(),
        cam / (cam.max() + 1e-9)
    )


@app.post("/tasks/{name}/explain")
async def explain(name: str, file: UploadFile = File(...)):
    if name not in TASKS:
        raise HTTPException(404, "Unknown task")

    data = await file.read()

    if not data:
        raise HTTPException(400, "Empty image")

    task = TASKS[name]
    size = task["meta"]["image_size"]

    try:
        image = tf.io.decode_image(
            data, channels=3, expand_animations=False
        )
        x = tf.image.resize(image, [size, size])[None]
    except (tf.errors.OpError, ValueError):
        raise HTTPException(400, "Invalid image")

    p, cam = gradcam(task["model"], x)

    heat = tf.image.resize(
        cam[None, :, :, None], [size, size]
    ).numpy()[0, :, :, 0]

    colours = np.stack([
        np.clip(3 * heat, 0, 1),
        np.clip(3 * heat - 1, 0, 1),
        np.clip(3 * heat - 2, 0, 1)
    ], axis=-1)

    overlay = (
        0.55 * x[0].numpy() + 0.45 * 255 * colours
    ).astype("uint8")

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")

    return {
        "task": name,
        "top_class": task["meta"]["classes"][int(p.argmax())],
        "probability": float(p.max()),
        "cam": cam.tolist(),
        "overlay_png": base64.b64encode(
            buffer.getvalue()
        ).decode()
    }