"""
app.py
======
AI Sentiment Analysis Tool
--------------------------
Tech stack : Python · scikit-learn · Gradio
Model      : TF-IDF Vectorizer + Logistic Regression
Persistence: Model trained once and saved as model.pkl via joblib

Usage:
    python app.py          # starts the web server (default port 8000)
    PORT=9000 python app.py  # custom port
"""

import os
import re
import sys
import string
from datetime import datetime

import joblib
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MODEL_PATH = "model.pkl"
MAX_CHARS  = 5000
MIN_CHARS  = 3

# Training dataset (positive = 1, negative = 0)
DATASET = [
    ("I absolutely love this product, it's amazing!", 1),
    ("This is the best day of my life!", 1),
    ("I feel so happy and grateful today.", 1),
    ("What a fantastic experience, I'm thrilled!", 1),
    ("The service was excellent and the staff were very friendly.", 1),
    ("I'm so excited about this opportunity!", 1),
    ("This movie was absolutely wonderful.", 1),
    ("Great job! You did outstanding work.", 1),
    ("The food was delicious and the atmosphere was perfect.", 1),
    ("I had an incredible time, would definitely recommend.", 1),
    ("Everything went smoothly, I'm very satisfied.", 1),
    ("Brilliant performance, truly breathtaking.", 1),
    ("The quality is top-notch, I couldn't be happier.", 1),
    ("Really enjoyed every moment of this experience.", 1),
    ("Superb customer service, they went above and beyond.", 1),
    ("This is exactly what I was looking for, perfect!", 1),
    ("Wonderful, I'm so pleased with the result.", 1),
    ("Very impressed with the quality and speed of delivery.", 1),
    ("The app works flawlessly, I love it!", 1),
    ("Such a heartwarming story, brought tears to my eyes.", 1),
    ("Incredible value for money, highly recommend.", 1),
    ("The team was professional, kind, and very efficient.", 1),
    ("This product changed my life for the better.", 1),
    ("Absolutely stunning design, exceeded all expectations.", 1),
    ("I'm genuinely impressed, this is remarkable work.", 1),
    ("I love this product so much!", 1),
    ("Best purchase I have ever made.", 1),
    ("So happy with the results, thank you!", 1),
    ("Exceeded every single expectation I had.", 1),
    ("Would 100% recommend to anyone looking for quality.", 1),
    ("This is terrible, I'm very disappointed.", 0),
    ("Worst experience of my life, never again.", 0),
    ("I hate this product, it's completely useless.", 0),
    ("The service was awful and the staff were rude.", 0),
    ("I'm so frustrated and angry right now.", 0),
    ("This movie was boring and a waste of time.", 0),
    ("Horrible quality, broke after one day of use.", 0),
    ("Very disappointing, nothing worked as described.", 0),
    ("Terrible customer support, they ignored my complaint.", 0),
    ("I regret buying this, complete waste of money.", 0),
    ("Disgusting food, made me feel sick.", 0),
    ("This is the worst app I have ever used.", 0),
    ("Extremely poor quality, very unhappy with this purchase.", 0),
    ("The delivery was late and the item arrived damaged.", 0),
    ("I'm totally unsatisfied, this did not meet expectations.", 0),
    ("Dreadful experience, I would not recommend this to anyone.", 0),
    ("The product is faulty and the refund process is painful.", 0),
    ("Absolutely awful, nothing went right.", 0),
    ("Lousy service, rude staff, and overpriced.", 0),
    ("I'm so fed up with these constant problems.", 0),
    ("This company has the worst customer service I've seen.", 0),
    ("Broken on arrival, total rubbish.", 0),
    ("Such a letdown, I expected so much more.", 0),
    ("I'm deeply unhappy with how this was handled.", 0),
    ("Poor workmanship, fell apart within a week.", 0),
    ("This is the worst experience I have ever had.", 0),
    ("Not worth the money at all, very bad quality.", 0),
    ("I would never buy this again, completely useless.", 0),
    ("Zero stars if I could, absolutely dreadful.", 0),
    ("Scam product, does not work as advertised at all.", 0),
]


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def preprocess(text: str) -> str:
    """
    Normalize raw text before model inference.
    1. Lowercase all characters
    2. Remove all punctuation
    3. Collapse consecutive whitespace to a single space
    4. Strip leading / trailing whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════
# MODEL — LOAD OR AUTO-TRAIN
# ═══════════════════════════════════════════════════════════════════════

def _build_pipeline():
    """Construct the sklearn TF-IDF → Logistic Regression pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=5000,
            sublinear_tf=True,    # log normalization
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.5,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def train_and_save() -> object:
    print("[INFO] Training model from scratch ...")
    texts  = [preprocess(t) for t, _ in DATASET]
    labels = [lbl for _, lbl in DATASET]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Model Accuracy: {acc:.2f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")
    return pipeline


def load_model() -> object:
    """Load model from disk if it exists, otherwise train and save it."""
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading pre-trained model from '{MODEL_PATH}' ...")
        pipeline = joblib.load(MODEL_PATH)
        print("[INFO] Model ready.")
        return pipeline
    print(f"[WARN] '{MODEL_PATH}' not found — auto-training ...")
    return train_and_save()


# Load once at startup — no retraining on every run
try:
    model = load_model()
except Exception as exc:
    print(f"[CRITICAL] Cannot load or train model: {exc}", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def predict_sentiment(text: str) -> tuple:
    """Classify text sentiment and return an HTML result card + stats."""
    word_count = len(text.split()) if text else 0
    char_count = len(text) if text else 0
    stats_html = _stats_html(word_count, char_count)

    # Validation
    if not text or not text.strip():
        return _error_card("Please enter some text to analyze."), stats_html
    if len(text.strip()) < MIN_CHARS:
        return _error_card("Input too short. Please enter a complete sentence."), stats_html
    if len(text) > MAX_CHARS:
        return _error_card(f"Input too long — maximum {MAX_CHARS:,} characters allowed."), stats_html

    # Inference
    try:
        cleaned       = preprocess(text)
        predicted_cls = int(model.predict([cleaned])[0])
        probabilities = model.predict_proba([cleaned])[0]
        confidence    = float(probabilities[predicted_cls]) * 100
        label  = "Positive" if predicted_cls == 1 else "Negative"
        emoji  = "😊" if predicted_cls == 1 else "😞"
        is_pos = predicted_cls == 1
        return _result_card(label, emoji, confidence, is_pos), stats_html
    except Exception as exc:
        return _error_card(f"An unexpected error occurred: {exc}"), stats_html


# ═══════════════════════════════════════════════════════════════════════
# HTML CARD BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def _result_card(label, emoji, confidence, is_positive):
    """Render a color-coded result card showing Sentiment + Confidence Score."""
    if is_positive:
        bg, border, badge_bg = "linear-gradient(135deg,#d1fae5,#a7f3d0)", "#10b981", "#059669"
        bar_clr, text_clr    = "#10b981", "#064e3b"
    else:
        bg, border, badge_bg = "linear-gradient(135deg,#fee2e2,#fecaca)", "#ef4444", "#dc2626"
        bar_clr, text_clr    = "#ef4444", "#7f1d1d"
    bar_pct = round(confidence, 1)

    return f"""
<div style="background:{bg};border:2px solid {border};border-radius:16px;
            padding:26px 28px 22px;font-family:'Segoe UI',system-ui,sans-serif;">
    <div style="font-size:.72rem;font-weight:700;letter-spacing:1.8px;
                text-transform:uppercase;color:{badge_bg};margin-bottom:6px;">
        Analysis Result
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <span style="font-size:.97rem;font-weight:600;color:{text_clr};opacity:.75;">Sentiment:</span>
        <span style="font-size:1.75rem;font-weight:800;color:{text_clr};">{label}</span>
        <span style="font-size:1.9rem;">{emoji}</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
        <span style="font-size:.85rem;font-weight:600;color:{text_clr};opacity:.75;">Confidence Score:</span>
        <span style="font-size:1.05rem;font-weight:800;color:{text_clr};">{confidence:.2f}%</span>
    </div>
    <div style="background:rgba(0,0,0,.10);border-radius:999px;height:10px;overflow:hidden;">
        <div style="width:{bar_pct}%;height:100%;background:{bar_clr};border-radius:999px;
                    transition:width .5s ease;"></div>
    </div>
</div>"""


def _error_card(message):
    """Render a warning card for validation errors."""
    return f"""
<div style="background:#fefce8;border:2px solid #f59e0b;border-radius:16px;
            padding:20px 24px;font-family:'Segoe UI',system-ui,sans-serif;
            color:#78350f;display:flex;align-items:center;gap:12px;">
    <span style="font-size:1.8rem;">⚠️</span>
    <span style="font-size:.97rem;font-weight:500;">{message}</span>
</div>"""


def _stats_html(words, chars):
    """Show live word/character count below the input box."""
    return f"""
<div style="display:flex;gap:16px;font-size:.82rem;color:#6b7280;padding:6px 2px;">
    <span>📝 <strong>{words}</strong> word{'s' if words!=1 else ''}</span>
    <span>🔤 <strong>{chars}</strong> / {MAX_CHARS:,} chars</span>
</div>"""


def _empty_result():
    """Placeholder shown before the first analysis."""
    return """
<div style="border:2px dashed #d1d5db;border-radius:16px;padding:28px;
            text-align:center;color:#9ca3af;font-size:.95rem;">
    <div style="font-size:2rem;margin-bottom:8px;">🔍</div>
    Your result will appear here after analysis.
</div>"""


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE INPUTS
# ═══════════════════════════════════════════════════════════════════════

EXAMPLES = [
    ["I love this product"],
    ["This is terrible"],
    ["The customer service was outstanding and very helpful."],
    ["I regret buying this — total waste of money."],
    ["What a beautiful day, feeling so grateful and happy!"],
    ["The quality is poor and it broke within a week."],
    ["Highly recommend this to everyone, it exceeded my expectations!"],
    ["I'm frustrated — nothing works as advertised."],
]


# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
button.primary {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    transition: opacity .2s, transform .15s !important;
}
button.primary:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
footer { display: none !important; }
"""


# ═══════════════════════════════════════════════════════════════════════
# GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════

with gr.Blocks(title="AI Sentiment Analysis Tool") as demo:

    # Header
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);border-radius:18px;
                padding:30px 36px 26px;margin-bottom:24px;color:white;
                font-family:'Segoe UI',system-ui,sans-serif;">
        <h1 style="margin:0 0 10px;font-size:1.85rem;font-weight:800;">
            🧠 AI Sentiment Analysis Tool
        </h1>
        <p style="margin:0;font-size:.97rem;opacity:.88;line-height:1.6;max-width:600px;">
            Type any sentence or paragraph and the model instantly classifies it as
            <strong>Positive</strong> or <strong>Negative</strong>, with a precise confidence score.
            Uses <em>TF-IDF</em> for feature extraction and <em>Logistic Regression</em>
            for classification — a fast, interpretable ML pipeline built with <strong>scikit-learn</strong>.
        </p>
    </div>""")

    # Input + Result
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            text_input    = gr.Textbox(label="Enter your text",
                                       placeholder='"I love this product"  or  "This was terrible"',
                                       lines=5, max_lines=12)
            stats_display = gr.HTML(value=_stats_html(0, 0))
        with gr.Column(scale=4):
            result_display = gr.HTML(value=_empty_result())

    # Buttons
    with gr.Row():
        clear_btn   = gr.Button("🗑️  Clear",             variant="secondary", size="sm")
        analyze_btn = gr.Button("🔍  Analyze Sentiment",  variant="primary",   size="lg")

    gr.HTML("<hr style='margin:20px 0;border-color:#e5e7eb;'>")

    # Examples
    gr.Markdown("### 💡 Try an example")
    gr.Examples(examples=EXAMPLES, inputs=text_input, label="", examples_per_page=8)

    gr.HTML("<hr style='margin:20px 0;border-color:#e5e7eb;'>")

    # How it works
    gr.HTML("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:22px 26px;
                font-family:'Segoe UI',system-ui,sans-serif;">
        <div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:14px;">⚙️ How it works</div>
        <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;">
            <div style="min-width:26px;height:26px;background:#6d28d9;color:white;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;">1</div>
            <div style="font-size:.9rem;color:#475569;line-height:1.6;padding-top:3px;">
                Your text is <strong style="color:#1e293b;">lowercased</strong> and
                <strong style="color:#1e293b;">punctuation is removed</strong>.
            </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;">
            <div style="min-width:26px;height:26px;background:#6d28d9;color:white;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;">2</div>
            <div style="font-size:.9rem;color:#475569;line-height:1.6;padding-top:3px;">
                Cleaned text → <strong style="color:#1e293b;">TF-IDF features</strong> (numerical word-importance vectors).
            </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;">
            <div style="min-width:26px;height:26px;background:#6d28d9;color:white;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;">3</div>
            <div style="font-size:.9rem;color:#475569;line-height:1.6;padding-top:3px;">
                <strong style="color:#1e293b;">Logistic Regression</strong> predicts
                <strong style="color:#059669;">Positive</strong> or <strong style="color:#dc2626;">Negative</strong>.
            </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:12px;">
            <div style="min-width:26px;height:26px;background:#6d28d9;color:white;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;">4</div>
            <div style="font-size:.9rem;color:#475569;line-height:1.6;padding-top:3px;">
                <strong style="color:#1e293b;">Confidence score</strong> comes from the model's predicted probability.
            </div>
        </div>
    </div>""")

    # Event wiring
    def run_analysis(text):
        result_html, stats_html = predict_sentiment(text)
        return result_html, stats_html

    text_input.change(
        fn=lambda t: _stats_html(len(t.split()) if t else 0, len(t) if t else 0),
        inputs=text_input, outputs=stats_display)

    analyze_btn.click(fn=run_analysis, inputs=text_input, outputs=[result_display, stats_display])
    text_input.submit(fn=run_analysis, inputs=text_input, outputs=[result_display, stats_display])
    clear_btn.click(fn=lambda: ("", _empty_result(), _stats_html(0, 0)),
                    inputs=None, outputs=[text_input, result_display, stats_display])


# ═══════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, show_error=True)