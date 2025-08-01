from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from utils.signal_utils import live_signal_generator
from utils.coherence_utils import transform, coherence
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
    step_counter = 0  # Track steps for coherence calculation
    wavelet_coherence_data = None  # Store latest coherence data
    
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
    signal_procedures = [None for _ in range(N_SIGNALS)]
    procedure_names = ['filter', 'normalize', 'smooth', 'detrend', 'amplify', 'clip', 'offset', 'none']
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

        # Compute pairwise cosine similarities between FFT outputs
        # fft_mod shape: (N_SIGNALS, Nfft)
        distances = []
        cosine_similarities = []
        pair_indices = []
        for i in range(N_SIGNALS):
            for j in range(i + 1, N_SIGNALS):
                # Cosine similarity: 1 - cosine(fft_mod[i], fft_mod[j])
                a = fft_mod[i]
                b = fft_mod[j]
                dot = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                cosine_sim = float(dot) / (float(norm_a) * float(norm_b) + 1e-8)
                # Convert similarity to a 'distance' (higher = more different)
                dist = float(1.0 - cosine_sim)
                distances.append(dist)
                cosine_similarities.append(cosine_sim)
                pair_indices.append((i, j))
        # Every 10 steps, randomly assign a new procedure to one signal
        if step_counter % 10 == 0:
            idx = np.random.randint(0, N_SIGNALS)
            # Pick a random procedure
            proc = np.random.choice(procedure_names)
            signal_procedures[idx] = proc

        # Calculate wavelet coherence every 2 steps for highest similarity pair
        step_counter += 1
        if step_counter % 5 == 0:
            try:
                # Find pair with highest cosine similarity
                max_sim_idx = np.argmax(cosine_similarities)
                i, j = pair_indices[max_sim_idx]
                
                # Get the signals for the most similar pair
                signal1 = buffer[i, :]
                signal2 = buffer[j, :]
                
                # Calculate wavelet transform for both signals
                coeffs1, freqs = transform(signal1, FS, highest=FS/2, lowest=2.0, nfreqs=200)
                coeffs2, _ = transform(signal2, FS, highest=FS/2, lowest=2.0, nfreqs=200)

                # Calculate wavelet coherence
                coh, _, xwt = coherence(coeffs1, coeffs2, freqs)

                # Extract phase from cross-wavelet transform
                phases = np.angle(xwt)  # Phase in radians

                # Calculate cone of influence (COI)
                # For Morlet wavelet, COI is usually sqrt(2) * scale (or period) from each edge
                # Here, we approximate scale as 1/frequency
                N_time = coh.shape[1] if hasattr(coh, 'shape') else len(coh[0])
                coi = np.zeros((len(freqs), N_time))
                for t in range(N_time):
                    edge_dist = min(t, N_time - t - 1)
                    for k, f in enumerate(freqs):
                        scale = 1.0 / f
                        # COI mask: 1 if inside cone, 0 if outside
                        coi[k, t] = 1 if edge_dist >= np.sqrt(2) * scale else 0

                # Store coherence data (convert to serializable format)
                wavelet_coherence_data = {
                    "coherence": coh.tolist() if hasattr(coh, 'tolist') else coh,
                    "phases": phases.tolist() if hasattr(phases, 'tolist') else phases,
                    "freqs": freqs.tolist() if hasattr(freqs, 'tolist') else freqs,
                    "coi": coi.tolist(),
                    "pair": [int(i), int(j)],
                    "pair_labels": [chr(65 + i), chr(65 + j)],  # Convert to A, B, C, etc.
                    "similarity": float(cosine_similarities[max_sim_idx])
                }
            except Exception as e:
                print(f"Error calculating wavelet coherence: {e}")
                wavelet_coherence_data = None

        data = {
            "signals": buffer.tolist(),
            "fft": fft_mod.tolist(),
            "distances": distances,
            "wavelet_coherence": wavelet_coherence_data,
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

