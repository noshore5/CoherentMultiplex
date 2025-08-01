from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from utils.signal_utils import live_signal_generator
import numpy as np
import json
import time
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

N_SIGNALS = 8
FS = 100
CHUNK_SIZE = 100


def signal_stream():
    N_SIGNALS = 8
    FS = 100
    MAX_POINTS = 256
    frame_rate = 1
    gen = live_signal_generator(N_SIGNALS, FS, frame_rate)
    # Initialize buffer with 256 points
    init_signals = []
    points_needed = MAX_POINTS
    while points_needed > 0:
        chunk = next(gen)
        signals = chunk['signals']
        if signals.shape[1] > points_needed:
            signals = signals[:, :points_needed]
        init_signals.append(signals)
        points_needed -= signals.shape[1]
    buffer = np.concatenate(init_signals, axis=1)
    while True:
        chunk = next(gen)
        signals = chunk['signals']
        fft = chunk['fft']
        # Ensure FFT is non-empty and modulus
        if isinstance(fft, np.ndarray):
            # If complex, take modulus
            if np.iscomplexobj(fft):
                fft_mod = np.abs(fft)
            else:
                fft_mod = fft
        else:
            fft_mod = np.array(fft)
            if np.iscomplexobj(fft_mod):
                fft_mod = np.abs(fft_mod)
        buffer = np.concatenate([buffer, signals], axis=1)
        if buffer.shape[1] > MAX_POINTS:
            buffer = buffer[:, -MAX_POINTS:]

        # Compute pairwise Euclidean distances between FFT outputs
        # fft_mod shape: (N_SIGNALS, Nfft)
        distances = []
        for i in range(N_SIGNALS):
            for j in range(i + 1, N_SIGNALS):
                dist = float(np.linalg.norm(fft_mod[i] - fft_mod[j]))
                distances.append(dist)

        data = {
            "signals": buffer.tolist(),
            "fft": fft_mod.tolist(),
            "distances": distances,
            "timestamp": time.time()
        }
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(1 / frame_rate)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/coherentmultiplex.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

@app.get("/stream")
async def stream():
    return StreamingResponse(signal_stream(), media_type="text/event-stream")

