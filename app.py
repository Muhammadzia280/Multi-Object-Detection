import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionX AI | YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

  .stApp {
    background: linear-gradient(
        135deg,
        #fce7f3 0%,
        #f9a8d4 45%,
        #f472b6 100%
    );
    color: #3b0a25;
}

    footer {
        visibility: hidden;
    }

    header {
        background: transparent;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        background: linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa,
            #22d3ee
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .class-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 18px 10px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        transition: 0.3s;
    }

    .class-card:hover {
        transform: translateY(-4px);
        border-color: #60a5fa;
    }

    .class-icon {
        font-size: 30px;
    }

    .class-name {
        color: #e2e8f0;
        font-weight: 700;
        margin-top: 5px;
    }

    .class-id {
        color: #64748b;
        font-size: 12px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #f8fafc;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .info-box {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(96, 165, 250, 0.20);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
    }

    .status-online {
        display: inline-block;
        padding: 7px 15px;
        border-radius: 30px;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
        font-weight: 700;
        font-size: 14px;
    }

    .download-text {
        color: #38bdf8;
        font-weight: 700;
        font-size: 18px;
    }

    section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ef4444 0%,
        #dc2626 50%,
        #991b1b 100%
    );
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">🤖 VisionX AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'YOLOv8 Multi-Object Detection &nbsp;•&nbsp; '
    'Real-Time Computer Vision'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div style="text-align:center;">'
    '<span class="status-online">● AI MODEL ONLINE</span>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;">
        <h1 style="font-size:28px;">🧠 VisionX</h1>
        <p style="color:#94a3b8;">
            AI Detection Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 Detection Mode")

mode = st.sidebar.radio(
    "Choose an option",
    [
        "🖼 Image Detection",
        "🎥 Video Detection",
        "📷 Real-Time Webcam"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Model Information")

st.sidebar.write("**Model:** YOLOv8 Nano")
st.sidebar.write("**Classes:** 8")
st.sidebar.write("**Input:** Image / Video / Camera")
st.sidebar.write("**Confidence:** 0.25")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="text-align:center;color:#64748b;font-size:12px;">
        Built with YOLOv8 + OpenCV + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CLASS CARDS
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Detection Classes</div>',
    unsafe_allow_html=True
)

classes = [
    ("👤", "Person", "0"),
    ("🚗", "Car", "1"),
    ("🏍️", "Bike", "2"),
    ("🚌", "Bus", "3"),
    ("🐕", "Dog", "4"),
    ("🐈", "Cat", "5"),
    ("🟢", "Mask", "6"),
    ("🔴", "No_Mask", "7")
]

cols = st.columns(8)

for col, item in zip(cols, classes):

    icon, name, class_id = item

    with col:

        st.markdown(
            f"""
            <div class="class-card">
                <div class="class-icon">{icon}</div>
                <div class="class-name">{name}</div>
                <div class="class-id">Class {class_id}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("---")


# =========================================================
# DRAW DETECTIONS
# =========================================================

def draw_detections(frame, results):

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            if class_name == "Mask":

                color = (0, 255, 0)

            elif class_name == "No_Mask":

                color = (0, 0, 255)

            else:

                color = (255, 0, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return frame


# =========================================================
# IMAGE DETECTION
# =========================================================

if mode == "🖼 Image Detection":

    st.markdown(
        '<div class="section-title">🖼 Image Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
            Upload an image and let the YOLOv8 model detect
            Person, Car, Bike, Bus, Dog, Cat, Mask and No_Mask.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 📷 Original")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### 🔍 Detection")

            if st.button(
                "🚀 Detect Objects",
                use_container_width=True
            ):

                with st.spinner("AI is detecting objects..."):

                    results = model(
                        image,
                        conf=0.25,
                        verbose=False
                    )

                    frame = results[0].orig_img.copy()

                    frame = draw_detections(
                        frame,
                        results
                    )

                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                st.image(
                    frame,
                    use_container_width=True
                )

                st.success(
                    "✅ Detection completed successfully!"
                )


# =========================================================
# VIDEO DETECTION
# =========================================================

elif mode == "🎥 Video Detection":

    st.markdown(
        '<div class="section-title">🎥 Video Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
            Upload a video to create a complete AI detection video.
            The processed video can be downloaded and shared on LinkedIn.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(
            uploaded_video.read()
        )

        temp_input.close()

        if st.button(
            "🎬 Start AI Video Detection",
            use_container_width=True
        ):

            cap = cv2.VideoCapture(
                temp_input.name
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            if fps <= 0:
                fps = 30

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_file.name

            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            current_frame = 0

            progress = st.progress(0)

            frame_placeholder = st.empty()

            status_placeholder = st.empty()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model(
                    frame,
                    conf=0.25,
                    verbose=False
                )

                frame = draw_detections(
                    frame,
                    results
                )

                out.write(frame)

                display_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame_placeholder.image(
                    display_frame,
                    channels="RGB",
                    use_container_width=True
                )

                current_frame += 1

                if total_frames > 0:

                    progress_value = min(
                        current_frame / total_frames,
                        1.0
                    )

                    progress.progress(
                        progress_value
                    )

                    status_placeholder.write(
                        f"Processing frame "
                        f"{current_frame}/{total_frames}"
                    )

            cap.release()

            out.release()

            progress.progress(1.0)

            status_placeholder.empty()

            frame_placeholder.empty()

            st.success(
                "✅ Video detection completed!"
            )

            st.markdown("### 🎬 Detection Result")

            st.video(
                output_path
            )

            with open(
                output_path,
                "rb"
            ) as file:

                video_bytes = file.read()

            st.markdown(
                '<div class="download-text">'
                '⬇️ Your AI detection video is ready'
                '</div>',
                unsafe_allow_html=True
            )

            st.download_button(
                label="⬇️ Download Detection Video",
                data=video_bytes,
                file_name="YOLOv8_Multi_Object_Detection.mp4",
                mime="video/mp4",
                use_container_width=True
            )


# =========================================================
# REAL-TIME WEBCAM
# =========================================================

elif mode == "📷 Real-Time Webcam":

    st.markdown(
        '<div class="section-title">📷 Real-Time Webcam</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
            Start your browser camera and perform real-time
            object detection using your trained YOLOv8 model.
        </div>
        """,
        unsafe_allow_html=True
    )

    try:

        from streamlit_webrtc import (
            webrtc_streamer,
            VideoProcessorBase
        )

        class YOLOVideoProcessor(
            VideoProcessorBase
        ):

            def recv(self, frame):

                img = frame.to_ndarray(
                    format="bgr24"
                )

                results = model(
                    img,
                    conf=0.25,
                    verbose=False
                )

                img = draw_detections(
                    img,
                    results
                )

                return frame.from_ndarray(
                    img,
                    format="bgr24"
                )

        webrtc_streamer(
            key="visionx-webcam",
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

    except Exception as e:

        st.error(
            f"Webcam error: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
    ">
        <b>VisionX AI</b><br>
        YOLOv8 Multi-Object Detection System<br>
        <small>
            Person • Car • Bike • Bus • Dog • Cat • Mask • No_Mask
        </small>
    </div>
    """,
    unsafe_allow_html=True
)