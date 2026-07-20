"""VehicleGen AI – Smart Vehicle Information Assistant — text classification using TensorFlow/Keras."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st


st.set_page_config(page_title="VehicleGen AI – Smart Vehicle Information Assistant", page_icon="🔧", layout="wide")

# Custom responsive CSS styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apply globally */
    .stApp {
        background: linear-gradient(135deg, #09090e 0%, #111126 50%, #05050b 100%) !important;
        color: #f1f5f9 !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
    }
    
    /* Sleek glowing headers */
    h1 {
        font-weight: 800 !important;
        background: linear-gradient(to right, #ffffff, #a5f3fc, #00f2fe) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: 0px 0px 30px rgba(0, 242, 254, 0.15) !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 8px !important;
    }
    
    h2, h3 {
        color: #00f2fe !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    
    /* Glassmorphic Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(6, 6, 12, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    
    /* Card design targeting Streamlit's native bordered container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 28px !important;
        margin: 20px 0px !important;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        transition: transform 0.25s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(0, 242, 254, 0.4) !important;
        box-shadow: 0 15px 45px rgba(0, 242, 254, 0.2) !important;
    }
    
    /* Premium input controls */
    textarea, input, div[data-baseweb="textarea"], div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    
    textarea:focus, input:focus, div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within {
        border-color: #00ffd0 !important;
        box-shadow: 0 0 12px rgba(0, 255, 208, 0.25) !important;
    }
    
    /* Premium button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0072ff 0%, #00f2fe 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
        font-family: 'Outfit', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.45) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #00ffd0 !important;
        text-shadow: 0 0 15px rgba(0, 255, 208, 0.2) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #94a3b8 !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 18px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

LABELS = [
    "Battery / starter system",
    "Brake safety issue",
    "Engine overheating",
    "Tyre or suspension issue",
    "Routine maintenance",
    "Fuel-efficiency issue",
]

TRAINING_DATA = [
    ("car clicks but will not start", 0), ("battery light is on", 0),
    ("engine cranks slowly in morning", 0), ("car needs jump start", 0),
    ("starter makes clicking noise", 0), ("dashboard lights are dim", 0),
    ("brake pedal feels soft", 1), ("car makes noise while braking", 1),
    ("brake warning light is on", 1), ("vehicle pulls when braking", 1),
    ("brakes are not stopping properly", 1), ("burning smell from brakes", 1),
    ("engine temperature is very high", 2), ("car is overheating", 2),
    ("coolant is leaking", 2), ("steam coming from bonnet", 2),
    ("temperature warning light", 2), ("radiator fan is not working", 2),
    ("tyre pressure warning", 3), ("car shakes at high speed", 3),
    ("uneven tyre wear", 3), ("steering wheel vibrates", 3),
    ("suspension makes knocking sound", 3), ("car pulls to one side", 3),
    ("when should I change engine oil", 4), ("regular service requirements", 4),
    ("need maintenance for my car", 4), ("how often replace air filter", 4),
    ("car service schedule", 4), ("check coolant and wiper fluid", 4),
    ("car mileage has dropped", 5), ("using too much fuel", 5),
    ("how can I improve fuel efficiency", 5), ("poor average mileage", 5),
    ("fuel consumption is high", 5), ("car gives low kilometres per litre", 5),
]


@st.cache_resource(show_spinner="Training the Keras model for the first time…")
def train_model():
    # Importing TensorFlow is slow on some Windows computers. Loading it here
    # lets the responsive Streamlit interface appear immediately.
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    texts = [item[0] for item in TRAINING_DATA]
    targets = np.array([item[1] for item in TRAINING_DATA], dtype=np.int32)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=600, output_mode="int", output_sequence_length=12
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(8))
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        tf.keras.layers.Embedding(input_dim=600, output_dim=16, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    training_inputs = tf.constant([[text] for text in texts])
    model.fit(training_inputs, targets, epochs=150, verbose=0)
    return model


def advice(category: str) -> str:
    messages = {
        "Battery / starter system": "Check battery terminals for corrosion and arrange a battery/charging-system test. If the car will not start, avoid repeated start attempts.",
        "Brake safety issue": "Treat this as urgent. If braking is weak, noisy, or a brake warning appears, stop driving and contact a qualified mechanic.",
        "Engine overheating": "Stop in a safe place, switch off the engine, and allow it to cool. Do not open a hot radiator cap. Check for coolant leaks only after cooling.",
        "Tyre or suspension issue": "Check tyre pressure and visible tyre damage. Book an inspection for vibration, pulling, uneven wear, or suspension noises.",
        "Routine maintenance": "Follow the vehicle owner’s manual for exact intervals. Check engine oil, tyres, coolant, brakes, filters, lights, and battery condition.",
        "Fuel-efficiency issue": "Keep tyres correctly inflated, avoid harsh acceleration, and service filters and engine oil on schedule. A sudden mileage drop should be inspected.",
    }
    return messages[category]


def get_gemini_key() -> str | None:
    """Read a private local/Cloud secret without showing it in the web app."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except FileNotFoundError:
        return None


def generate_ai_explanation(question: str, category: str, confidence: float) -> str | None:
    """Use Gemini to explain the Keras prediction in natural language."""
    key = get_gemini_key()
    if not key:
        return None

    prompt = f"""You are VehicleGen AI, a safety-first smart vehicle information assistant.
The user wrote: {question}
A Keras text-classification model predicted: {category} ({confidence:.1f}% confidence).

Explain this prediction in simple language. Give 3 practical checks, an urgency level,
and clearly say when a qualified mechanic is needed. Never claim a certain diagnosis.
For brakes, steering, smoke, overheating, or fuel leaks, prioritize stopping the vehicle safely."""

    # Try new google-genai SDK first, then fall back to google-generativeai
    try:
        from google import genai
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
    except Exception:
        pass

    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=key)
        model_legacy = genai_legacy.GenerativeModel("gemini-2.0-flash")
        response = model_legacy.generate_content(prompt)
        return response.text
    except Exception:
        return None


st.title("🚗 VehicleGen AI – Smart Vehicle Information Assistant")
st.caption("A hybrid Generative AI and machine-learning project for automobile concerns.")

with st.sidebar:
    st.header("About this project")
    st.write("Keras predicts the issue category. Gemini turns the prediction into clear, natural-language guidance.")
    if get_gemini_key():
        st.success("Gemini Generative AI is connected.")
    else:
        st.info("Keras prediction works now. Add a Gemini key in Streamlit secrets to enable Generative AI explanations.")
    st.warning("For brakes, smoke, steering difficulty, overheating, or fuel leaks: stop driving and seek professional help.")

st.info("Try: “My car makes a clicking sound and will not start.”")
question = st.text_area("Describe the car issue", placeholder="Example: My car is using too much fuel and the mileage has dropped.", height=130)

if st.button("Predict issue category", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please describe the car issue first.")
    else:
        import tensorflow as tf

        model = train_model()
        probabilities = model.predict(tf.constant([[question]]), verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index]) * 100
        category = LABELS[best_index]

        # Results metrics card
        with st.container(border=True):
            left, right = st.columns(2)
            left.metric("Predicted category", category)
            right.metric("Model confidence", f"{confidence:.1f}%")

        # Recommendation card
        with st.container(border=True):
            st.subheader("Recommended next step")
            st.write(advice(category))

        # Gemini AI explanation card
        with st.container(border=True):
            st.subheader("🤖 Gemini AI explanation")
            try:
                ai_response = generate_ai_explanation(question, category, confidence)
                if ai_response:
                    st.write(ai_response)
                else:
                    st.info("Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` to enable the Generative AI explanation.")
            except Exception:
                st.warning("The Keras prediction is ready, but Gemini could not be reached. Check the API key and internet connection.")
        st.caption("This is a small educational model trained on example text, not a replacement for professional diagnosis.")
