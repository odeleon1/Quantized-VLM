# Legacy (retired code, not maintained)

- `main.py`: the original PyQt5 desktop application, retired in Phase 11 when the project moved to the FastAPI plus React web interface. It is kept only for reference.
- `pipeline.py`: the Phase 7 standalone camera to inference loop. Its `run_inference` and `load_model` were superseded by the live copies in `backend/app/services/eval.py`.

Neither file is imported by the running server. They are archived here and are not kept working against the current code.
