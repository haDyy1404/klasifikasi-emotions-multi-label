import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                            classification_report)
from sklearn.preprocessing import MultiLabelBinarizer
import re
from streamlit_option_menu import option_menu

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Klasifikasi Emosi",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #e6e6fa 0%, #d8bfd8 100%); padding: 2rem;
        border-radius: 10px; margin: 1rem 0; text-align: center; color: #4a4a4a;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .emotion-tag {
        display: inline-block; background: linear-gradient(45deg, #dda0dd, #d8bfd8);
        color: #4a4a4a; padding: 0.5rem 1rem; margin: 0.25rem; border-radius: 20px;
        font-weight: bold; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton > button {
        background: linear-gradient(90deg, #e6e6fa 0%, #d8bfd8 100%); color: #4a4a4a;
        border: none; border-radius: 25px; padding: 0.5rem 2rem; font-weight: bold;
        transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        background: linear-gradient(90deg, #d8bfd8 0%, #dda0dd 100%);
    }
</style>
""", unsafe_allow_html=True)

# DEFINISI & FUNGSI UTILITAS
EMOTION_LABELS = {
    0: "kekaguman", 1: "kesenangan", 2: "marah", 3: "kesal", 4: "persetujuan", 5: "peduli",
    6: "bingung", 7: "penasaran", 8: "keinginan", 9: "kecewa", 10: "ketidaksetujuan",
    11: "jijik", 12: "malu", 13: "semangat", 14: "takut", 15: "terima kasih", 16: "duka",
    17: "bahagia", 18: "cinta", 19: "cemas", 20: "optimis", 21: "bangga", 22: "menyadari",
    23: "lega", 24: "penyesalan", 25: "sedih", 26: "terkejut", 27: "netral"
}

EMOTION_EMOJIS = {
    "kekaguman": "🤩", "kesenangan": "😄", "marah": "😠", "kesal": "😒", "persetujuan": "👍",
    "peduli": "🤗", "bingung": "🤔", "penasaran": "🧐", "keinginan": "😍", "kecewa": "😞",
    "ketidaksetujuan": "👎", "jijik": "🤢", "malu": "😳", "semangat": "🔥", "takut": "😨",
    "terima kasih": "🙏", "duka": "😢", "bahagia": "😊", "cinta": "❤️", "cemas": "😟",
    "optimis": "💪", "bangga": "😎", "menyadari": "💡", "lega": "😌", "penyesalan": "😔",
    "sedih": "😭", "terkejut": "😮", "netral": "😐"
}

MODEL_FILES = {
    "Model 1: LR 2e-6 BS 16": "Model 1 LR 2e-6 BS 16.pt",
    "Model 2: LR 2e-6 BS 32": "Model 2 LR 2e-6 BS 32.pt",
    "Model 3: LR 4e-6 BS 16": "Model 3 LR 4e-6 BS 16.pt",
    "Model 4: LR 4e-6 BS 32": "Model 4 LR 4e-6 BS 32.pt",
    "Model 5: LR 5e-5 BS 16": "Model 5 LR 5e-5 BS 16.pt",
    "Model 6: LR 5e-5 BS 32": "Model 6 LR 5e-5 BS 32.pt",
}

MODEL_ICONS = {
    "Model 1: LR 2e-6 BS 16": "🏅", "Model 2: LR 2e-6 BS 32": "🏅", "Model 3: LR 4e-6 BS 16": "🏅",
    "Model 4: LR 4e-6 BS 32": "🏅", "Model 5: LR 5e-5 BS 16": "🏅", "Model 6: LR 5e-5 BS 32": "🏅",
}

THRESHOLD = 0.5

def parse_labels(label_str):
    if pd.isna(label_str): return []
    if isinstance(label_str, str):
        return [int(x.strip()) for x in label_str.split(',') if x.strip().isdigit()]
    return [int(label_str)]

def labels_to_text_with_emoji(labels):
    result = []
    for label in labels:
        emotion_name = EMOTION_LABELS.get(label, f"Unknown_{label}")
        emoji = EMOTION_EMOJIS.get(emotion_name, "❓")
        result.append(f"{emoji} {emotion_name.capitalize()}")
    return result

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model_and_tokenizer(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan!")
        return None, None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        model = AutoModelForSequenceClassification.from_pretrained(
            'indobenchmark/indobert-base-p1', num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        st.sidebar.success(f"Model `{model_path}` berhasil dimuat! ✅")
        return model, tokenizer, device
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model '{model_path}': {e}")
        return None, None, None

def format_model_option(model_name):
    icon = MODEL_ICONS.get(model_name, "🤖")
    return f"{icon} {model_name}"

def model_selector():
    st.sidebar.title("Opsi Model")
    selected_model_name = st.sidebar.selectbox(
        "🤖 **Pilih Model**", options=list(MODEL_FILES.keys()),
        format_func=format_model_option, help="Pilih model yang ingin Anda gunakan."
    )
    if selected_model_name:
        return MODEL_FILES[selected_model_name]
    return None

def predict_real_emotion(text, model, tokenizer, device):
    cleaned_text = preprocess_text(text)
    inputs = tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits)
    
    predictions = (probabilities > THRESHOLD).squeeze().cpu().numpy()
    predicted_indices = np.where(predictions == 1)[0]
    
    results = [{'label': idx, 'confidence': probabilities[0, idx].item()} for idx in predicted_indices]
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results

def evaluate_real_model(df, model, tokenizer, device):
    y_true = [parse_labels(labels) for labels in df['label']]
    y_pred = []
    texts = df['text_translated'].tolist()
    progress_bar = st.progress(0, text="Mengevaluasi data...")
    for i, text in enumerate(texts):
        results = predict_real_emotion(text, model, tokenizer, device)
        y_pred.append([res['label'] for res in results])
        progress_bar.progress((i + 1) / len(texts), text=f"Data ke-{i+1}/{len(texts)}")
    progress_bar.empty()
    mlb = MultiLabelBinarizer(classes=list(EMOTION_LABELS.keys()))
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    report = classification_report(
        y_true_bin, y_pred_bin, target_names=list(EMOTION_LABELS.values()),
        zero_division=0, output_dict=True
    )
    
    return {
        'exact_match': accuracy_score(y_true_bin, y_pred_bin),
        'f1_macro': f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
        'classification_report': report
    }

def home_page():
    st.markdown('<div class="main-header"><h1>🎭 Sistem Klasifikasi Emosi Multi-Label</h1><p>Analisis emosi menggunakan Deep Learning untuk teks bahasa Indonesia.</p></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Training Data", "43,398")
    col2.metric("🧪 Test Data", "5,427")
    col3.metric("✅ Validation Data", "5,426") 
    col4.metric("🏷️ Emotion Labels", "28")
    st.markdown("---")
    st.markdown("### 🎭 Kategori Emosi")
    emotions_text = ""
    for i, (label_id, emotion_name) in enumerate(EMOTION_LABELS.items()):
        if i > 0 and i % 7 == 0: emotions_text += "<br>"
        emoji = EMOTION_EMOJIS.get(emotion_name, "❓")
        emotions_text += f"<span style='margin-right: 20px; display: inline-block;'>{emoji} {emotion_name.capitalize()}</span>"
    st.markdown(emotions_text, unsafe_allow_html=True)

def classification_page():
    st.markdown('<div class="main-header"><h1>Klasifikasi Emosi</h1><p>Masukkan teks untuk menganalisis emosi</p></div>', unsafe_allow_html=True)
    selected_model_path = model_selector()
    if selected_model_path:
        model, tokenizer, device = load_model_and_tokenizer(selected_model_path)
        if model and tokenizer:
            user_text = st.text_area("Masukkan teks:", height=120, placeholder="Contoh: Saya merasa sangat bahagia hari ini...", key="main_text_input")
            if st.button("🚀 Prediksi Emosi", use_container_width=True, key="analyze_btn"):
                if user_text and user_text.strip():
                    with st.spinner(f"Menganalisis dengan model {selected_model_path}..."):
                        results = predict_real_emotion(user_text, model, tokenizer, device)
                        st.markdown("### 🎯 Hasil Prediksi")
                        if results:
                            predicted_labels = [res['label'] for res in results]
                            confidences = [res['confidence'] for res in results]
                            emotion_names_with_emoji = labels_to_text_with_emoji(predicted_labels)
                            emotion_html = ""
                            for i, emotion in enumerate(emotion_names_with_emoji):
                                emotion_html += f'<span class="emotion-tag">{emotion} ({confidences[i]:.1%})</span> '
                            st.markdown(emotion_html, unsafe_allow_html=True)
                        else:
                            st.warning("😐 Tidak ada emosi yang terdeteksi (di atas threshold).")
                else:
                    st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")

def evaluation_page():
    st.markdown('<div class="main-header"><h1>Evaluasi Kinerja Model</h1></div>', unsafe_allow_html=True)
    selected_model_path = model_selector()
    if selected_model_path:
        model, tokenizer, device = load_model_and_tokenizer(selected_model_path)
        if model and tokenizer:
            # Komponen untuk upload file
            uploaded_file = st.file_uploader("Upload file CSV:", type=['csv'], help="File harus memiliki kolom 'text_translated' dan 'label'")

            # Tombol "Mulai Evaluasi" selalu terlihat
            if st.button("🚀 Mulai Evaluasi", use_container_width=True):
                # Pengecekan dilakukan SETELAH tombol diklik
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    if 'text_translated' not in df.columns or 'label' not in df.columns:
                        st.error("File harus memiliki kolom 'text_translated' dan 'label'.")
                    else:
                        st.success(f"✅ Dataset dimuat: **{len(df):,}** baris.")
                        metrics = evaluate_real_model(df, model, tokenizer, device)
                        st.subheader("🏆 Hasil Evaluasi Model", divider="rainbow")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("🏆 Exact Match", f"{metrics['exact_match']:.2%}")
                        col2.metric("⚖️ F1-Score (Macro)", f"{metrics['f1_macro']:.2%}")
                        col3.metric("💎 Precision (Macro)", f"{metrics['precision_macro']:.2%}")
                        col4.metric("🧲 Recall (Macro)", f"{metrics['recall_macro']:.2%}")

                        with st.expander("Laporan Klasifikasi Lengkap per Emosi"):
                            report_dict = metrics['classification_report']
                            
                            header = f"{'Nama Emosi':<18}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}"
                            report_lines = [header, "-" * len(header)]
                            
                            summary_keys = ['accuracy', 'micro avg', 'macro avg', 'weighted avg', 'samples avg']
                            
                            for label, scores in report_dict.items():
                                if label not in summary_keys and isinstance(scores, dict):
                                    p = f"{scores['precision']:.2f}"
                                    r = f"{scores['recall']:.2f}"
                                    f1 = f"{scores['f1-score']:.2f}"
                                    s = str(int(scores['support']))
                                    report_lines.append(f"{label:<18}{p:>12}{r:>12}{f1:>12}{s:>12}")
                            
                            report_lines.append("")
                            
                            for key in summary_keys:
                                if key in report_dict:
                                    scores = report_dict[key]
                                    if key == 'accuracy':
                                        continue 
                                    elif isinstance(scores, dict):
                                        p = f"{scores['precision']:.2f}"
                                        r = f"{scores['recall']:.2f}"
                                        f1 = f"{scores['f1-score']:.2f}"
                                        s = str(int(scores['support']))
                                        report_lines.append(f"{key:<18}{p:>12}{r:>12}{f1:>12}{s:>12}")

                            formatted_report = "\n".join(report_lines)
                            st.code(formatted_report, language='text')
                        st.balloons()
                else:
                    # Peringatan jika tidak ada file yang diunggah
                    st.warning("⚠️ Silakan unggah file dataset CSV terlebih dahulu sebelum memulai evaluasi.")

def main():
    # Menggunakan option_menu horizontal di bagian atas halaman
    selected_page = option_menu(
        menu_title=None,
        options=["Beranda", "Klasifikasi", "Evaluasi"],
        icons=["house-door-fill", "search", "clipboard2-data-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#636363", "font-size": "18px"},
            "nav": {"justify-content": "center"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px 10px",
                "padding": "5px 15px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#d8bfd8"},
        }
    )
    
    # Logika untuk menampilkan halaman
    if selected_page == "Beranda":
        home_page()
    elif selected_page == "Klasifikasi":
        classification_page()
    elif selected_page == "Evaluasi":
        evaluation_page()

if __name__ == "__main__":
    main()