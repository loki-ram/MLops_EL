# MLops_EL

## Docker

Build the image locally:

```bash
docker build -t mlops_el:latest .
```

Run with docker (you should mount `mlruns` and `data` so models and data are available):

```bash
docker run --rm -p 8501:8501 \
	-v "${PWD}/mlruns:/app/mlruns:ro" \
	-v "${PWD}/data:/app/data:ro" \
	-e MODEL_BASE=/app/mlruns \
	mlops_el:latest
```

Or use docker-compose (recommended) to build and mount automatically:

```bash
docker-compose up --build
```

Notes:
- `MODEL_BASE` points to the mounted `mlruns` folder inside the container.
- The image installs Python packages from `requirements.txt`. If additional OS packages are required for certain wheels (e.g. `xgboost`), the Dockerfile installs lightweight build tools.