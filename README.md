# dl_final
A repository to create an app using fastapi and streamlit


## Live Class 7 app

TrashNet six-class classifier using pretrained MobileNetV2.

Training: head-trained (stopped early); existing learned weights were retained when training was stopped early for the live lesson. Fine-tuning was not completed.

Held-out test accuracy: 0.8391; macro-F1: 0.8095. Not-sure threshold: 0.89.

Notebook with the executed completion cell: notebooks/01_class7_completed.ipynb. The original notebook is preserved.

API: `python -m uvicorn app.api:app --host 0.0.0.0 --port 8000`

App: `python -m streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501`

Checks: `python tests/check_api.py`

Dataset: https://github.com/garythung/trashnet (MIT). Exact duplicate filtering only; near-duplicate/group leakage was not audited. One-object board-background data does not establish performance on mixed real-world waste. Grad-CAM is score attribution, not proof of material recognition.
