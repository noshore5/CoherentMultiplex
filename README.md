
# CoherentMultiplex



## Features

- Graph network of signal nodes
- Real-time market coherence visualization
- FastAPI backend for high performance
- Plotly-powered interactive charts
- Modular utility functions for data processing

## Getting Started

### 1. Create and Activate a Virtual Environment

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the Application

```sh
uvicorn main:app --reload
```

Visit [http://localhost:8000](http://localhost:8000) in your browser to view the live charts.

## Project Structure

- `main.py` — FastAPI app entry point
- `utils/coherence_utils.py` — Coherence computation using fCWT
- `templates/coherentmultiplex.html` — Main HTML template for charts
- `static/` — Static assets (CSS, JS, images)
- `requirements.txt` — Python dependencies
- `.gitignore` — Files and folders to exclude from version control

## Docker Support

Containerization is available. See the `Dockerfile` for instructions on building and running the app in a Docker container.

## Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

## License

This project is licensed under the MIT License.
