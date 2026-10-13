
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. ตั้งค่าหน้าเว็บไซต์
# ==========================================

st.set_page_config(
    page_title="Online Shopping Prediction",
    page_icon="🛒",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

# รองรับชื่อโฟลเดอร์ทั้งตัวพิมพ์เล็กและใหญ่
def find_folder(name):
    for folder in [name.lower(), name.capitalize()]:
        path = ROOT / folder
        if path.exists():
            return path
    return ROOT / name.lower()

MODEL_DIR = find_folder("models")
FIGURE_DIR = find_folder("figures")

MODEL_FILE = MODEL_DIR / "shopping_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

# ==========================================
# 2. โหลดโมเดล
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

if not MODEL_FILE.exists():
    st.error("ไม่พบไฟล์โมเดล shopping_model.joblib")
    st.info("กรุณากลับไปที่ VS Code แล้วรัน python train_model.py")
    st.stop()

model = load_model()

# ==========================================
# 3. หัวข้อเว็บไซต์
# ==========================================

st.title("🛒 ระบบทำนายการซื้อสินค้าบนเว็บไซต์")

st.markdown(
    """
    **Online Shopping Purchase Prediction**

    ระบบวิเคราะห์พฤติกรรมผู้เข้าชมเว็บไซต์
    และทำนายแนวโน้มการตัดสินใจซื้อสินค้า
    ด้วยเทคนิค Machine Learning
    """
)

st.divider()

# ==========================================
# 4. เมนูด้านข้าง
# ==========================================

st.sidebar.title("เมนูหลัก")

menu = st.sidebar.radio(
    "เลือกเมนู",
    [
        "ทำนายการซื้อสินค้า",
        "วิเคราะห์ข้อมูล",
        "ประสิทธิภาพโมเดล",
        "เกี่ยวกับโครงงาน"
    ]
)

# ==========================================
# 5. หน้าทำนายการซื้อสินค้า
# ==========================================

if menu == "ทำนายการซื้อสินค้า":

    st.header("ทำนายพฤติกรรมการซื้อสินค้า")

    st.write(
        "กรอกข้อมูลพฤติกรรมของผู้เข้าชมเว็บไซต์ "
        "จากนั้นกดปุ่มทำนายผล"
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            pages = st.number_input(
                "จำนวนหน้าเว็บไซต์ที่เข้าชม",
                min_value=0,
                value=8,
                step=1
            )

            minutes = st.number_input(
                "ระยะเวลาที่อยู่บนเว็บไซต์ (นาที)",
                min_value=0.0,
                value=7.0,
                step=0.5
            )

            items = st.number_input(
                "จำนวนสินค้าที่เข้าชม",
                min_value=0,
                value=10,
                step=1
            )

        with col2:

            discount = st.selectbox(
                "ลูกค้าเห็นโปรโมชั่นส่วนลดหรือไม่",
                ["yes", "no"],
                format_func=lambda x:
                    "เห็นส่วนลด" if x == "yes"
                    else "ไม่เห็นส่วนลด"
            )

            returning = st.selectbox(
                "ประเภทผู้เข้าชม",
                ["yes", "no"],
                format_func=lambda x:
                    "ลูกค้าเก่า" if x == "yes"
                    else "ลูกค้าใหม่"
            )

            device = st.selectbox(
                "อุปกรณ์ที่ใช้เข้าชมเว็บไซต์",
                ["mobile", "desktop", "tablet"],
                format_func=lambda x: {
                    "mobile": "โทรศัพท์มือถือ",
                    "desktop": "คอมพิวเตอร์",
                    "tablet": "แท็บเล็ต"
                }[x]
            )

        submitted = st.form_submit_button(
            "ทำนายการซื้อสินค้า",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "pages_viewed": pages,
            "session_minutes": minutes,
            "items_viewed": items,
            "discount_seen": discount,
            "returning_visitor": returning,
            "device": device
        }])

        prediction = int(
            model.predict(input_data)[0]
        )

        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        probability = float(
            probabilities[classes.index(1)]
        )

        st.divider()
        st.subheader("ผลการทำนาย")

        if prediction == 1:

            st.success(
                "ลูกค้ามีแนวโน้มตัดสินใจซื้อสินค้า"
            )

        else:

            st.warning(
                "ลูกค้ามีแนวโน้มไม่ซื้อสินค้า"
            )

        st.metric(
            "ความน่าจะเป็นที่จะซื้อสินค้า",
            f"{probability * 100:.2f}%"
        )

        st.progress(
            max(0.0, min(1.0, probability))
        )

        st.caption(
            "ผลการทำนายเป็นการประมาณจากข้อมูล "
            "และโมเดล Machine Learning "
            "ไม่ใช่การยืนยันพฤติกรรมจริงของลูกค้า"
        )

# ==========================================
# 6. หน้าวิเคราะห์ข้อมูล
# ==========================================

elif menu == "วิเคราะห์ข้อมูล":

    st.header("การวิเคราะห์พฤติกรรมลูกค้า")

    st.write(
        "แสดงกราฟจากการวิเคราะห์ชุดข้อมูล "
        "C05_online_purchase_classification.xlsx"
    )

    charts = [
        (
            "01_purchase_outcomes.png",
            "กราฟที่ 1: จำนวนลูกค้าที่ซื้อและไม่ซื้อสินค้า"
        ),
        (
            "02_purchase_by_device.png",
            "กราฟที่ 2: อัตราการซื้อแยกตามอุปกรณ์"
        ),
        (
            "03_session_duration.png",
            "กราฟที่ 3: ระยะเวลาเข้าชมเว็บไซต์"
        )
    ]

    for filename, title in charts:

        st.subheader(title)

        image_path = FIGURE_DIR / filename

        if image_path.exists():

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:

            st.info(
                f"ไม่พบกราฟ {filename} "
                "กรุณารัน train_model.py ก่อน"
            )

# ==========================================
# 7. หน้าประสิทธิภาพโมเดล
# ==========================================

elif menu == "ประสิทธิภาพโมเดล":

    st.header("ผลการประเมินโมเดล Machine Learning")

    if METRICS_FILE.exists():

        with open(
            METRICS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            metrics = json.load(f)

        st.success(
            f"โมเดลที่เลือกใช้งาน: {metrics['best_model']}"
        )

        results = pd.DataFrame(metrics["results"])

        st.subheader("ตารางเปรียบเทียบโมเดล")

        display_results = results.drop(
            columns=["confusion_matrix"],
            errors="ignore"
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("เปรียบเทียบ Accuracy และ F1-score")

        chart_data = results.set_index("model")[
            ["accuracy", "f1"]
        ]

        st.bar_chart(chart_data)

        st.subheader("Confusion Matrix")

        selected_model = st.selectbox(
            "เลือกโมเดลที่ต้องการดูผล",
            results["model"].tolist()
        )

        selected_result = next(
            r for r in metrics["results"]
            if r["model"] == selected_model
        )

        matrix = pd.DataFrame(
            selected_result["confusion_matrix"],
            index=["จริง: ไม่ซื้อ", "จริง: ซื้อ"],
            columns=["ทำนาย: ไม่ซื้อ", "ทำนาย: ซื้อ"]
        )

        st.dataframe(
            matrix,
            use_container_width=True
        )

        st.markdown("""
        ### ความหมายของค่าประเมินโมเดล

        **Accuracy:** สัดส่วนการทำนายที่ถูกต้องทั้งหมด

        **Precision:** เมื่อโมเดลทำนายว่าซื้อสินค้า
        มีสัดส่วนที่ซื้อจริงเท่าใด

        **Recall:** โมเดลสามารถค้นหาลูกค้าที่ซื้อจริง
        ได้มากเพียงใด

        **F1-score:** ค่าที่สะท้อนความสมดุลระหว่าง
        Precision และ Recall
        """)

    else:

        st.warning("ไม่พบไฟล์ metrics.json")

# ==========================================
# 8. หน้าเกี่ยวกับโครงงาน
# ==========================================

elif menu == "เกี่ยวกับโครงงาน":

    st.header("เกี่ยวกับโครงงาน")

    st.markdown("""
    ### ชื่อโครงงาน
    การทำนายการซื้อสินค้าบนเว็บไซต์
    (Online Shopping Purchase Prediction)

    ### วัตถุประสงค์
    1. เพื่อศึกษาพฤติกรรมผู้เข้าชมเว็บไซต์
    2. เพื่อวิเคราะห์ปัจจัยที่เกี่ยวข้องกับการซื้อสินค้า
    3. เพื่อพัฒนาโมเดล Machine Learning
    4. เพื่อสร้างระบบทำนายผ่านเว็บไซต์

    ### ข้อมูลที่ใช้
    C05_online_purchase_classification.xlsx

    ### โมเดล Machine Learning
    - Logistic Regression
    - Random Forest

    ### เครื่องมือ
    - Python
    - Visual Studio Code
    - Pandas
    - Scikit-learn
    - Streamlit
    - GitHub

    ### ประโยชน์ที่คาดว่าจะได้รับ
    ช่วยให้ธุรกิจออนไลน์เข้าใจพฤติกรรมลูกค้า
    และนำผลการวิเคราะห์ไปประกอบการวางแผนการตลาด
    เพื่อเพิ่มโอกาสในการขายสินค้า
    """)

st.divider()

st.caption(
    "Online Shopping Purchase Prediction | "
    "Machine Learning Project"
)
