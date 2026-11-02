import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Segmentasi Warna HSV",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #eaeaea !important;
    }

    .stApp h1, .stApp h2, .stApp h3,
    .stApp [data-testid="stSidebar"] h1,
    .stApp [data-testid="stSidebar"] h2,
    .stApp [data-testid="stSidebar"] h3 {
        color: #9C2007 !important;
        font-weight: 600;
    }

    .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp .stMarkdown, .stApp .stMarkdown p,
    .stApp [data-testid="stSidebar"] p,
    .stApp [data-testid="stSidebar"] span,
    .stApp [data-testid="stSidebar"] label,
    .stApp [data-testid="stSidebar"] .stMarkdown p {
        color: #9C2007 !important;
    }

    .stSlider label, .stSlider p,
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #9C2007 !important;
    }

    .stApp .stCaption, .stApp small,
    .stApp [data-testid="stCaptionContainer"] {
        color: #9C2007 !important;
    }

    .stApp hr {
        border-color: #d0d0d0 !important;
    }

    .stApp [data-testid="stFileUploader"] label,
    .stApp [data-testid="stFileUploader"] span,
    .stApp [data-testid="stFileUploader"] p,
    .stApp [data-testid="stFileUploader"] small {
        color: #9C2007 !important;
    }

    .stApp .streamlit-expanderHeader, 
    .stApp .streamlit-expanderHeader p,
    .stApp details summary span {
        color: #9C2007 !important;
    }

    .stApp table, .stApp th, .stApp td {
        color: #9C2007 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎨 Segmentasi Warna (Color Segmentation)")
st.markdown(
    "Ekstrak objek dari citra berdasarkan rentang warna **HSV** secara interaktif."
)
st.divider()

with st.sidebar:
    st.header("⚙️ Panel Kontrol")

    st.subheader("📁 Unggah Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("🔽 Batas Bawah (Lower Bound)")

    lower_h = st.slider("Lower Hue", min_value=0, max_value=179, value=0, step=1)
    lower_s = st.slider("Lower Saturation", min_value=0, max_value=255, value=50, step=1)
    lower_v = st.slider("Lower Value", min_value=0, max_value=255, value=50, step=1)

    st.divider()

    st.subheader("🔼 Batas Atas (Upper Bound)")

    upper_h = st.slider("Upper Hue", min_value=0, max_value=179, value=179, step=1)
    upper_s = st.slider("Upper Saturation", min_value=0, max_value=255, value=255, step=1)
    upper_v = st.slider("Upper Value", min_value=0, max_value=255, value=255, step=1)

    st.divider()

    st.caption(f"**Lower HSV:** ({lower_h}, {lower_s}, {lower_v})")
    st.caption(f"**Upper HSV:** ({upper_h}, {upper_s}, {upper_v})")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

    result_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🖼️ Citra Asli")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.markdown("### 🎭 Mask (Biner)")
        st.image(mask, use_container_width=True)

    with col3:
        st.markdown("### ✅ Hasil Segmentasi")
        st.image(result_rgb, use_container_width=True)

    st.divider()
    with st.expander("ℹ️ Informasi Citra", expanded=False):
        st.markdown(
            f"""
            | Properti | Nilai |
            |---|---|
            | **Dimensi** | {img_bgr.shape[1]} × {img_bgr.shape[0]} px |
            | **Channel** | {img_bgr.shape[2]} |
            | **Piksel Terdeteksi** | {int(np.sum(mask > 0)):,} dari {mask.size:,} ({np.sum(mask > 0) / mask.size * 100:.1f}%) |
            """
        )

else:
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 80px 20px;
            color: #999999;
        ">
            <p style="font-size: 48px; margin-bottom: 10px;">📷</p>
            <p style="font-size: 18px;">Unggah gambar melalui <b>Panel Kontrol</b> di sidebar untuk memulai segmentasi warna.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
