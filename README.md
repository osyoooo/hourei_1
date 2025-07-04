# Hourei API and App

This project provides a FastAPI backend and Streamlit front-end for searching and querying tax law data. The backend serves endpoints defined in `api.py`, while the Streamlit interface in `app.py` interacts with it.

## Installation

Install dependencies with `pip`:

```bash
pip install -r requirements.txt
```

## Usage

Start the API server:

```bash
python api.py
```

(or run via uvicorn: `uvicorn api:app --reload`)

Launch the Streamlit app:

```bash
streamlit run app.py
```

Ensure the API server is running before using the Streamlit interface.
