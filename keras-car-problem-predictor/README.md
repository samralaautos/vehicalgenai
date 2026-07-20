# VehicleGen AI – Smart Vehicle Information Assistant

A responsive hybrid Generative AI and machine-learning application. A TensorFlow/Keras model predicts the most likely automobile service category, then Gemini generates a clear, safety-first explanation and recommended next steps.

## Model workflow

1. A user describes an automobile concern.
2. Keras `TextVectorization` converts text into numerical sequences.
3. An Embedding layer and neural network classify it into one of six service categories.
4. The app displays the category, confidence score, and a safety-focused next step.
5. Gemini turns the classification result into a natural-language explanation.

## Categories

- Battery / starter system
- Brake safety issue
- Engine overheating
- Tyre or suspension issue
- Routine maintenance
- Fuel-efficiency issue

## Run locally

TensorFlow currently supports Python 3.9–3.12. Use Python 3.12 for this project.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Enable Gemini Generative AI

Create `.streamlit/secrets.toml` by copying `.streamlit/secrets.toml.example`, then add your Gemini API key:

```toml
GEMINI_API_KEY = "your_actual_key"
```

The Keras classifier still works without this key; the key enables the Gemini explanation section.

## Deployment

1. Push this folder to a public GitHub repository.
2. Create a free account on [Streamlit Community Cloud](https://share.streamlit.io) and connect your GitHub.
3. Deploy a new app selecting your repository, branch (`main`), and main file path (`app.py`).
4. Set your `GEMINI_API_KEY` in the application's **Secrets** settings under the advanced options on the Streamlit dashboard:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   ```
5. The included `runtime.txt` specifies Python 3.12, ensuring model dependencies load correctly on deployment.

## Note

This project combines Python, TensorFlow/Keras, Streamlit, and Gemini Generative AI in one application.
