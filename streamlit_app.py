
import streamlit as st
import numpy as np
from PIL import Image

# ตั้งค่าหน้าเว็บไซต์
st.set_page_config(
    page_title="ระบบวิเคราะห์ฟิล์มชีวภาพอัญชัน",
    page_icon="🦋",
    layout="wide"
)

# หัวข้อเว็บไซต์
st.title("🦋 ระบบวิเคราะห์ฟิล์มชีวภาพอัญชัน")
st.write("ระบบวิเคราะห์การเปลี่ยนแปลงสีของแผ่นฟิล์มเพื่อประเมินค่า pH")

st.divider()

# ส่วนอัปโหลดภาพ
st.header("📷 อัปโหลดภาพแผ่นฟิล์ม")

uploaded_file = st.file_uploader(
    "เลือกภาพถ่ายจากโทรศัพท์มือถือ",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # อ่านภาพ
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    height, width, _ = image_array.shape

    st.subheader("ภาพที่อัปโหลด")

    st.image(image, caption="ภาพแผ่นฟิล์มที่ใช้วิเคราะห์")

    st.write("ขนาดภาพ:", width, "x", height, "pixels")

    st.divider()

    # เลือกบริเวณแผ่นฟิล์ม
    st.header("🔍 เลือกบริเวณที่ต้องการวิเคราะห์")

    st.write("ปรับตำแหน่งกรอบให้ครอบคลุมบริเวณแผ่นฟิล์ม")

    x1, x2 = st.slider(
        "ตำแหน่งแนวนอน",
        min_value=0,
        max_value=width,
        value=(width // 4, width * 3 // 4)
    )

    y1, y2 = st.slider(
        "ตำแหน่งแนวตั้ง",
        min_value=0,
        max_value=height,
        value=(height // 4, height * 3 // 4)
    )

    if x2 > x1 and y2 > y1:

        # ตัดภาพเฉพาะบริเวณที่เลือก
        cropped = image_array[y1:y2, x1:x2]

        # คำนวณค่าเฉลี่ย RGB
        rgb_mean = np.mean(cropped, axis=(0, 1))

        st.subheader("ผลการวิเคราะห์สี")

        st.image(
            cropped,
            caption="บริเวณแผ่นฟิล์มที่เลือก"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Red (R)", round(rgb_mean[0], 2))
        col2.metric("Green (G)", round(rgb_mean[1], 2))
        col3.metric("Blue (B)", round(rgb_mean[2], 2))

        st.success("วิเคราะห์ค่าสีจากภาพเรียบร้อยแล้ว")

        st.info(
            "ค่า RGB เป็นข้อมูลสีเบื้องต้น "
            "ยังไม่ใช่ค่า pH โดยตรง "
            "ต้องนำไปเปรียบเทียบกับข้อมูลสีมาตรฐานจากการทดลองก่อน"
        )

else:
    st.info("กรุณาอัปโหลดภาพแผ่นฟิล์มเพื่อเริ่มการวิเคราะห์")
