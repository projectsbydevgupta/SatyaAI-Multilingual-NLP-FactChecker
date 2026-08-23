import streamlit as st
import string
import pickle
import re
import os
import json
import requests
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel
from groq import Groq

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SatyaAI — Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 SatyaAI — Fake News Detector")
st.caption("Cross-Lingual RAG-Based Fact-Checking System for Regional and Code-Mixed Indian News")

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _model_path(name, v2_name=None):
    """Return v2 path if it exists, else fall back to name."""
    v2 = os.path.join(MODELS_DIR, v2_name or name)
    base = os.path.join(MODELS_DIR, name)
    return v2 if os.path.exists(v2) else base

@st.cache_resource
def load_models():
    def load_pkl(path):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None

    return (
        load_pkl(_model_path("tfidf.pkl",   "tfidf_v2.pkl")),
        load_pkl(_model_path("logreg.pkl",  "logreg_v2.pkl")),
        load_pkl(_model_path("svm_v2.pkl")),
        load_pkl(_model_path("xlmr_classifier.pkl")),
    )


@st.cache_resource
def load_xlmr_backbone():
    name = "cardiffnlp/twitter-xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return tokenizer, model


tfidf, model_lr, model_svm, model_xlmr = load_models()

if tfidf is None:
    st.error(
        f"❌ **Models not found!**\n\n"
        f"Looking in: `{MODELS_DIR}`\n\n"
        f"Files found: `{os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'FOLDER MISSING'}`"
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
if "claim_input" not in st.session_state:
    st.session_state.claim_input = ""

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())


# Merged TRUE/REAL → success, FALSE/FAKE → error, rest → warning
_LABEL_MAP = {
    "TRUE":         ("✅ TRUE",          "success"),
    "REAL":         ("✅ REAL",          "success"),
    "FALSE":        ("🚨 FALSE",         "error"),
    "FAKE":         ("🚨 FAKE",          "error"),
    "UNVERIFIABLE": ("❓ UNVERIFIABLE",  "warning"),
    "ERROR":        ("⚠️ ERROR",         "error"),
}

def label_badge(label):
    return _LABEL_MAP.get(str(label).upper(), ("⚠️ MISLEADING", "warning"))


def get_xlmr_embedding(text, tokenizer, model):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state[:, 0, :].numpy()


# Common words that add noise to a news search query
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "that", "this", "it", "its", "as", "near", "into",
    "up", "out", "about", "than", "their", "they", "he", "she", "we"
}

def _extract_keywords(claim, n=8):
    """Return the n most informative words from the claim (skip stopwords)."""
    words = claim.translate(str.maketrans("", "", string.punctuation)).split()
    meaningful = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
    # Prefer longer words (more specific) and take first n
    meaningful.sort(key=len, reverse=True)
    return " ".join(meaningful[:n])


def get_evidence(claim, max_articles=5):
    try:
        GNEWS_KEY = st.secrets["GNEWS_API_KEY"]
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        keywords  = _extract_keywords(claim)
        url = (
            f"https://gnews.io/api/v4/search"
            f"?q={requests.utils.quote(keywords)}"
            f"&token={GNEWS_KEY}&lang=en"
            f"&max={max_articles}"
            f"&from={from_date}"
        )
        articles = requests.get(url, timeout=10).json().get("articles", [])
        if not articles:
            return None, "gnews_empty"
        lines = []
        for i, a in enumerate(articles, 1):
            lines.append(
                f"[{i}] {a['title']}\n"
                f"    📰 Source : {a['source']['name']}\n"
                f"    🔗 URL    : {a.get('url', 'N/A')}\n"
                f"    📝 Desc   : {a.get('description', 'No description')}"
            )
        return "\n\n".join(lines), "gnews"
    except Exception:
        return None, "gnews_error"


def groq_check(claim, lr_label, lr_conf, xlmr_label, xlmr_conf,
               svm_label, ens_label, evidence_text, evidence_source):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY_NLP"])

        ml_summary = f"- Logistic Regression : {lr_label} ({lr_conf}% confidence)\n"
        if xlmr_label != "N/A":
            ml_summary += f"- XLM-RoBERTa         : {xlmr_label} ({xlmr_conf}% confidence)\n"
        if svm_label  != "N/A":
            ml_summary += f"- SVM                 : {svm_label}\n"
        ml_summary += f"- Ensemble (best)     : {ens_label}"

        ev_instruction = (
            "Real-time GNews evidence is provided. Use it as PRIMARY source. Cite article source in reason."
            if evidence_source == "gnews"
            else (
                "No recent GNews evidence found. You MUST use your own training knowledge to verify this claim. "
                "Indian political events, protests, Parliament incidents, and police actions are well-documented "
                "in your training data. Do NOT default to UNVERIFIABLE just because GNews has no results — "
                "only use UNVERIFIABLE if you genuinely have NO knowledge of this event whatsoever."
            )
        )

        system_prompt = f"""You are SatyaAI, an expert Indian fact-checker with deep knowledge of Indian politics, \
news, and current events.
Rules:
1. {ev_instruction}
2. UNVERIFIABLE is a last resort — only use it if you have absolutely no information about this claim.
3. For specific claims about Indian protests, Parliament, government actions, or named organizations, \
you almost certainly have training knowledge — use it confidently.
4. Flag low ML confidence (below 60%) as suspicious.
5. Output ONLY valid JSON — no extra text.
Output format:
{{
  "verdict": "REAL or FAKE or UNVERIFIABLE",
  "confidence": <integer 0-100>,
  "reason": "<one clear sentence mentioning source>",
  "source_used": "GNews API" or "Groq Knowledge Base",
  "flags": ["<flag1>", "<flag2>"]
}}"""

        user_prompt = f"""
Claim: {claim}

ML Model Predictions:
{ml_summary}

News Evidence ({evidence_source}):
{evidence_text if evidence_text else "No recent news found. Use your own knowledge."}

Return your JSON verdict.
"""
        raw = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=400
        ).choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except Exception:
            return {"verdict": "UNVERIFIABLE", "confidence": 0,
                    "reason": raw, "source_used": "Groq Knowledge Base", "flags": []}
    except Exception as e:
        return {"verdict": "ERROR", "confidence": 0,
                "reason": str(e), "source_used": "None", "flags": []}


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("ℹ️ About SatyaAI")
    st.info(
        "📅 **Evidence Search Range**\n\n"
        "Searching news from the **last 30 days** (GNews free tier).\n\n"
        "⚠️ Older claims → Groq verifies using its own knowledge."
    )
    st.markdown("---")
    st.markdown("**🔧 Pipeline**")
    st.markdown(
        "1. 🌐 Input (English / Hindi / Hinglish)\n"
        "2. 🤖 LR + SVM → TF-IDF predictions\n"
        "3. 🧠 XLM-RoBERTa → Transformer prediction\n"
        "4. 🗳️ Soft Voting → Ensemble verdict\n"
        "5. 📰 GNews → Live evidence (last 30 days)\n"
        "6. 🤖 Groq → Final AI verdict\n"
        "7. ✅ REAL / FAKE / UNVERIFIABLE"
    )
    st.markdown("---")
    st.markdown("**⚙️ Loaded Models**")
    st.markdown(f"- TF-IDF      : {'✅ Loaded' if tfidf else '❌ Missing'}")
    st.markdown(f"- LR          : {'✅ Loaded' if model_lr else '❌ Missing'}")
    st.markdown(f"- SVM         : {'✅ Loaded' if model_svm else '❌ Missing'}")
    st.markdown(f"- XLM-RoBERTa : {'✅ Loaded' if model_xlmr else '❌ Missing'}")
    st.markdown("---")
    st.markdown("**🏷️ Label Guide**")
    st.success("✅ TRUE — Verified factual claim")
    st.error("🚨 FALSE — Factually incorrect")
    st.warning("⚠️ MISLEADING — Partially true / out of context")
    st.markdown("---")
    st.markdown("**🌐 Languages Supported**")
    st.markdown("🇬🇧 English  |  🇮🇳 Hindi  |  🔀 Hinglish")

# ═══════════════════════════════════════════════════════════════
# EXAMPLE CLAIMS
# ═══════════════════════════════════════════════════════════════
st.markdown("### 📋 Example Claims to Test")
st.caption("Click any button below to auto-fill the claim ↓")

EXAMPLES = [
    ("🇬🇧 English — TRUE",
     "India won the 2011 ICC Cricket World Cup defeating Sri Lanka by 6 wickets in the final held at Wankhede Stadium Mumbai."),
    ("🇬🇧 English — FALSE",
     "NASA scientists confirmed that drinking hot water with lemon cures COVID-19 completely within 24 hours."),
    ("🔀 Hinglish — FALSE",
     "COVID vaccine lagwane ke baad logo ke andar 5G chip dal di gayi hai jo unka dimag control karti hai."),
    ("🇮🇳 Hindi — MISLEADING",
     "वायरल वीडियो में राहुल गांधी भारत के विभाजन का समर्थन करते हुए दिख रहे हैं।"),
]

for label, text in EXAMPLES:
    col_a, col_b = st.columns([1.5, 5])
    with col_a:
        if st.button(f"Use → {label}", key=f"btn_{label}"):
            st.session_state.claim_input = text
            st.rerun()
    with col_b:
        st.code(text, language=None)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# MAIN INPUT
# ═══════════════════════════════════════════════════════════════
st.markdown("### 🔍 Enter a Claim to Verify")

claim = st.text_area(
    "Enter your claim here (English / Hindi / Hinglish)",
    value=st.session_state.claim_input,
    placeholder="e.g. India won the Cricket World Cup in 2011",
    height=100,
)

analyze_btn = st.button("🔍 Analyse Claim", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════
if analyze_btn:

    if not claim.strip():
        st.warning("⚠️ Please enter a claim to verify.")
        st.stop()

    # Step 1: TF-IDF
    vec = tfidf.transform([clean_text(claim)])

    # Step 2: Logistic Regression
    lr_label   = model_lr.predict(vec)[0]
    lr_proba   = model_lr.predict_proba(vec)[0]
    lr_conf    = round(max(lr_proba) * 100, 2)
    lr_classes = list(model_lr.classes_)

    # Step 3: SVM
    if model_svm:
        svm_label   = model_svm.predict(vec)[0]
        svm_proba   = model_svm.predict_proba(vec)[0]
        svm_conf    = round(max(svm_proba) * 100, 2)
        svm_classes = list(model_svm.classes_)
    else:
        svm_label, svm_conf, svm_proba, svm_classes = "N/A", 0, [], []

    # Step 4: XLM-RoBERTa
    if model_xlmr:
        with st.spinner("🔄 Running XLM-RoBERTa transformer..."):
            xlmr_tokenizer, xlmr_base = load_xlmr_backbone()
            emb = get_xlmr_embedding(claim, xlmr_tokenizer, xlmr_base)
        xlmr_label   = model_xlmr.predict(emb)[0]
        xlmr_proba   = model_xlmr.predict_proba(emb)[0]
        xlmr_conf    = round(max(xlmr_proba) * 100, 2)
        xlmr_classes = list(model_xlmr.classes_)
    else:
        xlmr_label, xlmr_conf, xlmr_proba, xlmr_classes = "N/A", 0, [], []

    # Step 5: Ensemble (soft voting)
    active_probas = [lr_proba]
    if model_svm:   active_probas.append(svm_proba)
    if model_xlmr:  active_probas.append(xlmr_proba)
    ensemble_proba = np.mean(active_probas, axis=0)
    ens_label      = lr_classes[int(np.argmax(ensemble_proba))]
    ens_conf       = round(max(ensemble_proba) * 100, 2)

    # Step 6: GNews Evidence
    with st.spinner("📰 Fetching live news evidence..."):
        evidence, evidence_source = get_evidence(claim)

    st.markdown("---")

    # ── ROW 1: LR | XLM-RoBERTa | Evidence ─────────────────────
    st.subheader("📊 ML Model Predictions")
    col_lr, col_xlmr, col_ev = st.columns([1, 1, 2])

    with col_lr:
        st.markdown("**🔵 Logistic Regression**")
        lr_text, lr_type = label_badge(lr_label)
        getattr(st, lr_type)(f"{lr_text}\n\n**Confidence: {lr_conf}%**")
        st.progress(int(lr_conf))
        with st.expander("Class Probabilities"):
            for cls, prob in zip(lr_classes, lr_proba):
                st.write(f"`{cls}`: {round(prob*100, 1)}%")
        if lr_conf < 60:
            st.caption("⚠️ Low confidence — model is unsure")

    with col_xlmr:
        st.markdown("**🟣 XLM-RoBERTa (Transformer)**")
        if not model_xlmr:
            st.info("XLM-RoBERTa model not found in models/ folder.")
        else:
            xlmr_text, xlmr_type = label_badge(xlmr_label)
            getattr(st, xlmr_type)(f"{xlmr_text}\n\n**Confidence: {xlmr_conf}%**")
            st.progress(int(xlmr_conf))
            with st.expander("Class Probabilities"):
                for cls, prob in zip(xlmr_classes, xlmr_proba):
                    st.write(f"`{cls}`: {round(prob*100, 1)}%")
            if xlmr_conf < 60:
                st.caption("⚠️ Low confidence — model is unsure")

    with col_ev:
        st.markdown("**📰 Live News Evidence**")
        st.caption("📅 Last 30 days  |  🔍 GNews API")
        if evidence and evidence_source == "gnews":
            st.success("✅ Evidence retrieved from **GNews API**")
            st.markdown("📡 **Source: GNews API — Real-time News**")
            st.text(evidence)
        else:
            st.warning("📭 No recent news found in GNews (last 30 days).")
            st.info("🧠 **Fallback: Groq Knowledge Base**\n\nGroq will verify using its own training knowledge.")

    # ── ROW 2: SVM | Ensemble ───────────────────────────────────
    st.markdown("---")
    col_svm, col_ens = st.columns(2)

    with col_svm:
        st.markdown("**🟠 SVM (Linear)**")
        if not model_svm:
            st.info("SVM model not found in models/ folder.")
        else:
            svm_text, svm_type = label_badge(svm_label)
            getattr(st, svm_type)(f"{svm_text}\n\n**Confidence: {svm_conf}%**")
            st.progress(int(svm_conf))
            with st.expander("Class Probabilities"):
                for cls, prob in zip(svm_classes, svm_proba):
                    st.write(f"`{cls}`: {round(prob*100, 1)}%")
            if svm_conf < 60:
                st.caption("⚠️ Low confidence — model is unsure")

    with col_ens:
        st.markdown("**🏆 Ensemble (LR + SVM + XLM-RoBERTa)**")
        ens_text, ens_type = label_badge(ens_label)
        getattr(st, ens_type)(f"{ens_text}\n\n**Ensemble Confidence: {ens_conf}%**")
        st.progress(int(ens_conf))
        with st.expander("Ensemble Class Probabilities"):
            for cls, prob in zip(lr_classes, ensemble_proba):
                st.write(f"`{cls}`: {round(prob*100, 1)}%")

    # ── GROQ FINAL VERDICT ──────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 Groq — Final AI Verdict")

    with st.spinner("🤖 Asking Groq for final verdict..."):
        groq_result = groq_check(
            claim=claim,
            lr_label=lr_label, lr_conf=lr_conf,
            xlmr_label=xlmr_label, xlmr_conf=xlmr_conf,
            svm_label=svm_label, ens_label=ens_label,
            evidence_text=evidence if evidence else "No recent news found.",
            evidence_source=evidence_source
        )

    verdict     = groq_result.get("verdict",     "UNVERIFIABLE")
    confidence  = groq_result.get("confidence",  0)
    reason      = groq_result.get("reason",      "No reason provided.")
    source_used = groq_result.get("source_used", "Groq Knowledge Base")
    flags       = groq_result.get("flags",       [])

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        v_text, v_type = label_badge(verdict)
        getattr(st, v_type)(f"## {v_text}")
        st.metric("AI Confidence", f"{confidence}%")
        st.progress(int(confidence))

    with col_v2:
        st.markdown("**📝 Reasoning:**")
        st.info(reason)
        if source_used == "GNews API":
            st.success("📡 **Verified using: GNews API** (Real-time news articles)")
        else:
            st.info("🧠 **Verified using: Groq Knowledge Base**\n\nNo recent GNews articles found.")
        if flags:
            st.markdown("**🚩 Flags:**")
            for flag in flags:
                st.markdown(f"- {flag}")

    # ── MODEL AGREEMENT TABLE ───────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Model Agreement Summary")

    st.table(pd.DataFrame({
        "Model"      : ["Logistic Regression", "XLM-RoBERTa", "SVM", "Ensemble", "Groq"],
        "Prediction" : [lr_label, xlmr_label, svm_label, ens_label, verdict],
        "Confidence" : [
            f"{lr_conf}%",
            f"{xlmr_conf}%" if xlmr_label != "N/A" else "—",
            f"{svm_conf}%"  if svm_label  != "N/A" else "—",
            f"{ens_conf}%",
            f"{confidence}%"
        ],
        "Source" : [
            "TF-IDF Features",
            "Transformer Embeddings",
            "TF-IDF Features",
            "Soft Voting (LR+SVM+XLMR)",
            source_used
        ]
    }))
    st.markdown("---")
    st.caption("🔬 SatyaAI — XLM-RoBERTa + TF-IDF (LR+SVM) + Groq openai/gpt-oss-120b | NLP Project 2025")