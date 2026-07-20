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


LOCAL_EXPLANATIONS = {
    "Battery / starter system": {
        "summary": "Your vehicle's symptoms strongly suggest a **battery or starter system issue**. This is one of the most common vehicle problems and is usually straightforward to diagnose and fix.",
        "checks": [
            "🔋 **Check the battery terminals** — Open the bonnet and look for white or blue powdery corrosion on the battery terminals. Clean them carefully with a dry cloth.",
            "💡 **Check dashboard warning lights** — A battery warning light (looks like a rectangle with + and − signs) means the alternator is not charging the battery while driving.",
            "🔑 **Listen when you turn the key** — A single click usually means a faulty starter motor. Rapid clicking usually means a flat battery. No sound at all may mean a blown fuse.",
        ],
        "urgency": "⚠️ **Medium urgency.** The vehicle may not restart once turned off. Do not turn off the engine until you reach a garage.",
        "mechanic": "Visit a mechanic if the battery is more than 3–4 years old, if jump-starting does not help, or if the battery warning light stays on while driving.",
    },
    "Brake safety issue": {
        "summary": "Your description suggests a **brake system issue**. This is a **safety-critical** problem that must be taken seriously. Brakes are the most important safety system on your vehicle.",
        "checks": [
            "🛑 **Check brake pedal feel** — If the pedal sinks to the floor or feels spongy, this may indicate low brake fluid or a fluid leak. Check the brake fluid reservoir under the bonnet.",
            "👂 **Listen for noises** — Squealing usually means worn brake pads. Grinding means the pads are completely worn and metal is contacting the disc — stop driving immediately.",
            "🚗 **Check for pulling** — If the car pulls strongly to one side when braking, one brake caliper may be sticking or a brake pad may be unevenly worn.",
        ],
        "urgency": "🚨 **HIGH URGENCY — Safety Critical.** If braking ability is reduced, do not drive the vehicle. Call a mechanic or have the vehicle towed.",
        "mechanic": "See a qualified mechanic immediately if you hear grinding, the pedal goes to the floor, or the brake warning light is on. Do not delay brake repairs.",
    },
    "Engine overheating": {
        "summary": "Your vehicle appears to be experiencing **engine overheating**. This is a serious issue that can cause permanent and very expensive engine damage if not addressed immediately.",
        "checks": [
            "🌡️ **Watch the temperature gauge** — If the needle moves past the midpoint toward 'H' (Hot), pull over safely and switch off the engine. Do NOT open the radiator cap while hot.",
            "💧 **Check coolant level (when cold)** — Once the engine has cooled (30+ minutes), check the coolant reservoir. Low coolant is the most common cause of overheating.",
            "🌬️ **Check the radiator fan** — With the engine running, check if the cooling fan is spinning. A failed fan will cause overheating especially in traffic or at low speeds.",
        ],
        "urgency": "🚨 **HIGH URGENCY.** Pull over and stop the engine immediately if the temperature gauge is in the red or steam is visible. Continuing to drive will destroy the engine.",
        "mechanic": "Visit a mechanic before driving again. Common causes include a broken water pump, thermostat, radiator leak, or blown head gasket — all require professional repair.",
    },
    "Tyre or suspension issue": {
        "summary": "Your symptoms indicate a **tyre or suspension problem**. These issues affect vehicle safety, handling, and tyre lifespan, and should not be ignored.",
        "checks": [
            "🔄 **Check tyre pressure** — Use a tyre pressure gauge to check all four tyres including the spare. Incorrect pressure causes vibration, uneven wear, and poor handling.",
            "👁️ **Visually inspect tyres** — Look for bulges, cracks, nails, or uneven wear patterns. Uneven wear (more on one edge) usually indicates wheel alignment or suspension problems.",
            "🛞 **Test the suspension** — Press down firmly on each corner of the vehicle and release. It should bounce back once and stop. More bouncing means worn shock absorbers.",
        ],
        "urgency": "⚠️ **Medium urgency.** Tyre issues can become dangerous suddenly (blowout). Book an inspection within 1–2 days, sooner if you notice the car pulling or vibrating badly.",
        "mechanic": "See a mechanic for wheel alignment, balancing, or suspension inspection if vibration or pulling persists. Tyre replacement should be done professionally.",
    },
    "Routine maintenance": {
        "summary": "Your vehicle appears to be due for **routine maintenance**. Regular servicing is the single most important thing you can do to extend the life of your vehicle.",
        "checks": [
            "🛢️ **Check engine oil** — Pull the dipstick, wipe it clean, reinsert, and check again. Oil should be between MIN and MAX marks and appear amber/golden (not black and gritty).",
            "📅 **Review your service history** — Check when the last service was done. Most vehicles need a service every 10,000–15,000 km or once a year, whichever comes first.",
            "💡 **Check all fluid levels** — Check coolant, brake fluid, power steering fluid, and windscreen washer fluid. Top up with the correct fluid type as per the owner's manual.",
        ],
        "urgency": "✅ **Low urgency.** Routine maintenance is not an emergency, but neglecting it leads to costly repairs. Book a service within the next 2–4 weeks.",
        "mechanic": "A standard full service at a qualified garage includes oil change, filter replacement, fluid top-ups, brake inspection, and safety checks. Follow your manufacturer's schedule.",
    },
    "Fuel-efficiency issue": {
        "summary": "Your vehicle is experiencing a **fuel efficiency problem**. Poor fuel economy costs you money every day and often signals an underlying mechanical issue.",
        "checks": [
            "🔵 **Check tyre pressure** — Under-inflated tyres are the #1 cause of poor fuel economy. Even 10% below recommended pressure can increase fuel use by 2–3%.",
            "🔧 **Check the air filter** — A clogged air filter restricts airflow to the engine, forcing it to burn more fuel. Hold it up to the light — if you cannot see through it, replace it.",
            "⛽ **Review your driving habits** — Harsh acceleration, heavy braking, and idling all burn excess fuel. Smooth, steady driving at moderate speeds gives best economy.",
        ],
        "urgency": "ℹ️ **Low-Medium urgency.** A sudden, significant drop in fuel economy (not gradual) can indicate a faulty oxygen sensor, fuel injector issue, or other engine problem needing attention.",
        "mechanic": "See a mechanic if fuel economy has dropped suddenly (not gradually over years). Fault codes from an OBD-II scanner can quickly pinpoint the cause.",
    },
}


def generate_ai_explanation(question: str, category: str, confidence: float) -> str:
    """Use Gemini to explain the prediction; fall back to built-in explanations if unavailable."""
    key = get_gemini_key()
    if key:
        prompt = f"""You are VehicleGen AI, a safety-first smart vehicle information assistant.
The user wrote: {question}
A Keras text-classification model predicted: {category} ({confidence:.1f}% confidence).

Explain this prediction in simple language. Give 3 practical checks, an urgency level,
and clearly say when a qualified mechanic is needed. Never claim a certain diagnosis.
For brakes, steering, smoke, overheating, or fuel leaks, prioritize stopping the vehicle safely."""

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
            pass

    # Built-in fallback explanation — always works, no API needed
    data = LOCAL_EXPLANATIONS.get(category, {})
    if not data:
        return ""
    lines = [data["summary"], ""]
    lines.append("**🔍 3 Practical Checks:**")
    for check in data["checks"]:
        lines.append(f"- {check}")
    lines.append("")
    lines.append(data["urgency"])
    lines.append("")
    lines.append(f"**👨‍🔧 When to see a mechanic:** {data['mechanic']}")
    return "\n".join(lines)


st.title("🚗 VehicleGen AI – Smart Vehicle Information Assistant")
st.caption("A hybrid Generative AI and machine-learning project for automobile concerns.")

with st.sidebar:
    st.header("About this project")
    st.write("Keras predicts the issue category. Gemini turns the prediction into clear, natural-language guidance.")
    if get_gemini_key():
        st.success("Gemini Generative AI is connected.")
    else:
        st.info("Running with built-in AI explanations. Add a Gemini key to `.streamlit/secrets.toml` to enable live Gemini responses.")
    st.warning("For brakes, smoke, steering difficulty, overheating, or fuel leaks: stop driving and seek professional help.")

st.info("Try: \u201cMy car makes a clicking sound and will not start.\u201d")
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

        # AI explanation card — always shows rich content
        with st.container(border=True):
            st.subheader("🤖 AI Vehicle Explanation")
            ai_response = generate_ai_explanation(question, category, confidence)
            if ai_response:
                st.markdown(ai_response)
            else:
                st.info("Detailed explanation unavailable. Please check your connection.")
        st.caption("This is a small educational model trained on example text, not a replacement for professional diagnosis.")
