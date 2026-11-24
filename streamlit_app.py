import streamlit as st
import sqlite3
import numpy as np
import pandas as pd

from PIL import Image
from io import BytesIO
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error


# ==========================================
# 1. ตั้งค่าหน้าเว็บไซต์
# ==========================================

st.set_page_config(
    page_title="ระบบวิเคราะห์ฟิล์มชีวภาพอัญชัน",
    page_icon="🦋",
    layout="wide"
)

DB_NAME = "experiment_data.db"

st.title("🦋 ระบบวิเคราะห์ฟิล์มชีวภาพอัญชัน")
st.caption("ระบบบันทึกผลการทดลอง วิเคราะห์สี และติดตามการเปลี่ยนแปลงค่า pH")


# ==========================================
# 2. สร้างฐานข้อมูล SQLite
# ==========================================

def get_connection():
    return sqlite3.connect(DB_NAME)


def create_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_code TEXT NOT NULL,
                experiment_date TEXT NOT NULL,
                contact_hours REAL,
                ph_value REAL,
                notes TEXT,
                red REAL,
                green REAL,
                blue REAL,
                hue REAL,
                saturation REAL,
                value REAL,
                image BLOB
            )
        """)


create_database()


# ==========================================
# 3. ฟังก์ชันวิเคราะห์ภาพ RGB และ HSV
# ==========================================

def analyze_image(image, x1, x2, y1, y2):

    image = image.convert("RGB")
    image_array = np.array(image)

    # ตัดเฉพาะบริเวณที่ต้องการวิเคราะห์
    cropped = image_array[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError("กรุณาเลือกบริเวณภาพให้ถูกต้อง")

    # คำนวณ RGB เฉลี่ย
    rgb_mean = np.mean(cropped, axis=(0, 1))

    # แปลงภาพ RGB เป็น HSV ด้วย Pillow
    hsv_image = Image.fromarray(cropped).convert("HSV")
    hsv_array = np.array(hsv_image).astype(float)

    hsv_mean = np.mean(hsv_array, axis=(0, 1))

    # Pillow เก็บ Hue ในช่วง 0-255
    # แปลงเป็นองศา 0-360
    hue = hsv_mean[0] * 360 / 255

    # Saturation และ Value อยู่ในช่วง 0-100
    saturation = hsv_mean[1] * 100 / 255
    value = hsv_mean[2] * 100 / 255

    return {
        "red": float(rgb_mean[0]),
        "green": float(rgb_mean[1]),
        "blue": float(rgb_mean[2]),
        "hue": float(hue),
        "saturation": float(saturation),
        "value": float(value)
    }, cropped


# ==========================================
# 4. ฟังก์ชันบันทึกข้อมูลลงฐานข้อมูล
# ==========================================

def save_experiment(data):

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO experiments (
                sample_code,
                experiment_date,
                contact_hours,
                ph_value,
                notes,
                red,
                green,
                blue,
                hue,
                saturation,
                value,
                image
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["sample_code"],
            data["experiment_date"],
            data["contact_hours"],
            data["ph_value"],
            data["notes"],
            data["red"],
            data["green"],
            data["blue"],
            data["hue"],
            data["saturation"],
            data["value"],
            data["image"]
        ))


# ==========================================
# 5. ฟังก์ชันดึงข้อมูลจากฐานข้อมูล
# ==========================================

def load_data():

    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT
                id,
                sample_code,
                experiment_date,
                contact_hours,
                ph_value,
                notes,
                red,
                green,
                blue,
                hue,
                saturation,
                value
            FROM experiments
            ORDER BY experiment_date, id
        """, conn)

    return df


# ==========================================
# 6. เมนูหลักของเว็บไซต์
# ==========================================

menu = st.sidebar.radio(
    "📌 เมนูการใช้งาน",
    [
        "หน้าหลัก",
        "บันทึกผลการทดลอง",
        "ประวัติและกราฟ",
        "ดาวน์โหลดข้อมูล",
        "ประเมินค่า pH"
    ]
)


# ==========================================
# 7. หน้าหลัก
# ==========================================

if menu == "หน้าหลัก":

    st.header("ยินดีต้อนรับ")

    st.write("""
    เว็บไซต์นี้จัดทำขึ้นเพื่อบันทึกผลการทดลอง
    ฟิล์มชีวภาพจากสารสกัดอัญชันที่ใช้สัมผัสกับแหนม

    ระบบสามารถวิเคราะห์การเปลี่ยนแปลงสีของฟิล์ม
    ด้วยค่าสี RGB และ HSV
    พร้อมบันทึกผลการวัดค่า pH เพื่อใช้เปรียบเทียบ
    """)

    df = load_data()

    col1, col2, col3 = st.columns(3)

    col1.metric("จำนวนการทดลอง", len(df))

    if len(df) > 0:
        col2.metric(
            "ค่า pH ล่าสุด",
            "ยังไม่มีข้อมูล" if df["ph_value"].isna().all()
            else round(df["ph_value"].dropna().iloc[-1], 2)
        )
    else:
        col2.metric("ค่า pH ล่าสุด", "-")

    col3.metric(
        "จำนวนตัวอย่างที่มีภาพ",
        len(df)
    )

    st.info("เริ่มต้นใช้งานได้ที่เมนู 'บันทึกผลการทดลอง'")


# ==========================================
# 8. หน้าบันทึกผลการทดลอง
# ==========================================

elif menu == "บันทึกผลการทดลอง":

    st.header("📝 บันทึกผลการทดลอง")

    st.write("กรอกข้อมูลตัวอย่างและอัปโหลดภาพฟิล์มที่ต้องการวิเคราะห์")

    uploaded_file = st.file_uploader(
        "อัปโหลดภาพฟิล์มชีวภาพอัญชัน",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        st.subheader("ภาพที่อัปโหลด")
        st.image(image, use_container_width=True)

        width, height = image.size

        st.subheader("เลือกบริเวณฟิล์มที่ต้องการวิเคราะห์")

        st.caption(
            "ปรับกรอบให้ครอบคลุมเฉพาะบริเวณฟิล์ม "
            "และหลีกเลี่ยงพื้นหลังหรือส่วนที่มีเงา"
        )

        x1, x2 = st.slider(
            "เลือกตำแหน่งแนวนอน",
            min_value=0,
            max_value=width,
            value=(0, width)
        )

        y1, y2 = st.slider(
            "เลือกตำแหน่งแนวตั้ง",
            min_value=0,
            max_value=height,
            value=(0, height)
        )

        if x2 > x1 and y2 > y1:

            color_data, cropped = analyze_image(
                image, x1, x2, y1, y2
            )

            st.image(
                cropped,
                caption="บริเวณที่ใช้วิเคราะห์สี",
                use_container_width=True
            )

            st.subheader("ผลการวิเคราะห์สีเบื้องต้น")

            col1, col2, col3 = st.columns(3)

            col1.metric("Red", round(color_data["red"], 2))
            col2.metric("Green", round(color_data["green"], 2))
            col3.metric("Blue", round(color_data["blue"], 2))

            col4, col5, col6 = st.columns(3)

            col4.metric("Hue", round(color_data["hue"], 2))
            col5.metric("Saturation", round(color_data["saturation"], 2))
            col6.metric("Value", round(color_data["value"], 2))

            st.divider()

            st.subheader("ข้อมูลการทดลอง")

            with st.form("experiment_form"):

                sample_code = st.text_input(
                    "รหัสตัวอย่าง",
                    placeholder="เช่น NAEM-001"
                )

                experiment_date = st.date_input(
                    "วันที่ทดลอง",
                    value=date.today()
                )

                contact_hours = st.number_input(
                    "ระยะเวลาที่ฟิล์มสัมผัสกับแหนม (ชั่วโมง)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0
                )

                has_ph = st.checkbox(
                    "มีผลการวัดค่า pH จากกระดาษวัดแล้ว"
                )

                ph_value = None

                if has_ph:
                    ph_value = st.number_input(
                        "ค่า pH ที่อ่านได้",
                        min_value=0.0,
                        max_value=14.0,
                        value=7.0,
                        step=0.5
                    )

                notes = st.text_area(
                    "หมายเหตุเพิ่มเติม",
                    placeholder="เช่น สีฟิล์มเปลี่ยนเป็นสีม่วงอ่อน..."
                )

                submitted = st.form_submit_button(
                    "💾 บันทึกผลการทดลอง",
                    use_container_width=True
                )

                if submitted:

                    if not sample_code.strip():

                        st.error("กรุณากรอกรหัสตัวอย่าง")

                    elif x2 <= x1 or y2 <= y1:

                        st.error("กรุณาเลือกบริเวณภาพให้ถูกต้อง")

                    else:

                        data = {
                            "sample_code": sample_code.strip(),
                            "experiment_date": str(experiment_date),
                            "contact_hours": contact_hours,
                            "ph_value": ph_value,
                            "notes": notes,
                            **color_data,
                            "image": image_bytes
                        }

                        save_experiment(data)

                        st.success(
                            "บันทึกข้อมูลการทดลองเรียบร้อยแล้ว"
                        )

                        st.info(
                            "ข้อมูลถูกบันทึกลงฐานข้อมูล SQLite แล้ว"
                        )

    else:

        st.info("กรุณาอัปโหลดภาพก่อนเริ่มบันทึกผลการทดลอง")


# ==========================================
# 9. หน้าประวัติและกราฟ
# ==========================================

elif menu == "ประวัติและกราฟ":

    st.header("📊 ประวัติผลการทดลอง")

    df = load_data()

    if len(df) == 0:

        st.warning("ยังไม่มีข้อมูลการทดลอง")

    else:

        st.subheader("ตารางข้อมูลทั้งหมด")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("กราฟเปรียบเทียบค่า pH")

        ph_df = df.dropna(subset=["ph_value"])

        if len(ph_df) > 0:

            st.line_chart(
                ph_df.set_index("experiment_date")["ph_value"]
            )

        else:

            st.info("ยังไม่มีข้อมูล pH สำหรับสร้างกราฟ")

        st.divider()

        st.subheader("กราฟเปรียบเทียบค่าสีกับค่า pH")

        if len(ph_df) > 0:

            st.write("Hue เปรียบเทียบกับค่า pH")

            st.scatter_chart(
                ph_df,
                x="hue",
                y="ph_value"
            )

            st.write("Saturation เปรียบเทียบกับค่า pH")

            st.scatter_chart(
                ph_df,
                x="saturation",
                y="ph_value"
            )

            st.write("Value เปรียบเทียบกับค่า pH")

            st.scatter_chart(
                ph_df,
                x="value",
                y="ph_value"
            )

        else:

            st.info("ต้องมีข้อมูล pH ก่อนจึงจะสร้างกราฟนี้ได้")


# ==========================================
# 10. หน้าดาวน์โหลดข้อมูล CSV
# ==========================================

elif menu == "ดาวน์โหลดข้อมูล":

    st.header("📥 ดาวน์โหลดข้อมูลการทดลอง")

    df = load_data()

    if len(df) > 0:

        csv_data = df.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="ดาวน์โหลดข้อมูลเป็น CSV",
            data=csv_data,
            file_name="butterfly_pea_experiment.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.subheader("ตัวอย่างข้อมูล")

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.warning("ยังไม่มีข้อมูลสำหรับดาวน์โหลด")


# ==========================================
# 11. หน้าประเมินค่า pH
# ==========================================

elif menu == "ประเมินค่า pH":

    st.header("🧪 ระบบประเมินค่า pH จากสีของฟิล์ม")

    st.write("""
    ระบบนี้ใช้ความสัมพันธ์ระหว่างค่าสี HSV
    กับค่า pH ที่วัดได้จริงจากการทดลอง
    เพื่อสร้างแบบจำลอง Linear Regression
    """)

    df = load_data()

    calibration_df = df.dropna(
        subset=["ph_value", "hue", "saturation", "value"]
    ).copy()

    if len(calibration_df) < 15:

        st.warning(
            "ข้อมูลอ้างอิงยังไม่เพียงพอ "
            "ควรมีข้อมูลที่จับคู่ระหว่างภาพฟิล์มและค่า pH "
            "อย่างน้อย 15 ตัวอย่างก่อนทดลองสร้างแบบจำลอง"
        )

    elif calibration_df["ph_value"].nunique() < 5:

        st.warning(
            "ข้อมูลค่า pH ยังมีความหลากหลายไม่เพียงพอ "
            "ควรมีค่า pH อ้างอิงอย่างน้อย 5 ระดับ"
        )

    else:

        features = calibration_df[
            ["hue", "saturation", "value"]
        ]

        target = calibration_df["ph_value"]

        model = LinearRegression()

        # ประเมินความคลาดเคลื่อนด้วย Leave-One-Out
        loo = LeaveOneOut()

        predicted_cv = cross_val_predict(
            model,
            features,
            target,
            cv=loo
        )

        mae = mean_absolute_error(
            target,
            predicted_cv
        )

        model.fit(features, target)

        st.success("สร้างแบบจำลองจากข้อมูลทดลองแล้ว")

        st.metric(
            "ค่า MAE จากการตรวจสอบแบบ Leave-One-Out",
            round(mae, 3)
        )

        st.caption(
            "MAE คือค่าความคลาดเคลื่อนเฉลี่ยของการประเมิน "
            "ยิ่งต่ำยิ่งดี แต่ยังต้องตรวจสอบกับตัวอย่างใหม่"
        )

        st.divider()

        st.subheader("อัปโหลดภาพฟิล์มเพื่อทดลองประเมินค่า pH")

        prediction_file = st.file_uploader(
            "เลือกภาพฟิล์มสำหรับประเมิน",
            type=["jpg", "jpeg", "png"],
            key="prediction_image"
        )

        if prediction_file is not None:

            prediction_image = Image.open(
                prediction_file
            ).convert("RGB")

            st.image(
                prediction_image,
                caption="ภาพสำหรับประเมิน",
                use_container_width=True
            )

            width, height = prediction_image.size

            px1, px2 = st.slider(
                "เลือกแนวนอนของภาพ",
                0,
                width,
                (0, width),
                key="prediction_x"
            )

            py1, py2 = st.slider(
                "เลือกแนวตั้งของภาพ",
                0,
                height,
                (0, height),
                key="prediction_y"
            )

            if px2 > px1 and py2 > py1:

                prediction_colors, _ = analyze_image(
                    prediction_image,
                    px1, px2, py1, py2
                )

                new_data = pd.DataFrame([{
                    "hue": prediction_colors["hue"],
                    "saturation": prediction_colors["saturation"],
                    "value": prediction_colors["value"]
                }])

                estimated_ph = model.predict(new_data)[0]

                st.metric(
                    "ค่า pH ที่แบบจำลองประเมิน",
                    round(float(estimated_ph), 2)
                )

                st.warning("""
                ผลลัพธ์นี้เป็นเพียงค่าประเมินจากแบบจำลอง
                ไม่ใช่ค่าที่วัดได้จริง และไม่สามารถใช้ยืนยัน
                ความปลอดภัยหรือการเน่าเสียของอาหารได้
                ควรเปรียบเทียบกับวิธีวัด pH ที่เหมาะสมทุกครั้ง
                """)

                st.write(
                    "ความคลาดเคลื่อนเฉลี่ยจากการตรวจสอบแบบจำลอง:",
                    round(mae, 3),
                    "หน่วย pH"
                )
