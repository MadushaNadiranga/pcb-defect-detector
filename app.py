import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from io import BytesIO
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px



# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="img/pcb_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD CSS
# -----------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("pcb-defect-detector/exp12/weights/best.pt")

model = load_model()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    st.divider()
    st.header("📊 System Info")
    st.write(f"Model loaded: **PCB Defect Detector v1.0**")
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("Backend: **YOLOv8n (Ultralytics)**")
    st.write("Hardware: Running on CPU")

    st.divider()
    st.header("ℹ️ About")
    st.info("This tool detects defects in PCB boards using AI. \n\n"
            "Developed for **manufacturing quality inspection**.")

    st.markdown("---")
    st.caption("© 2025 PCB Manufacturing Solutions")

# -----------------------------
# APP HEADER
# -----------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">PCB Defect Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated defect detection with AI-powered vision</p>', unsafe_allow_html=True)

with col2:
    st.image("img/pcb_img.png", width=50)

# -----------------------------
# MAIN TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Image Analysis", 
    "Batch Processing", 
    "Real-Time Analysis",
    "Results History" 
  
])

# =============================
# TAB 1: SINGLE IMAGE ANALYSIS
# =============================
with tab1:
    st.subheader("Single PCB Analysis")
    uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"], key="single_upload")

    if uploaded_file is not None:
        try:
            col1, col2 = st.columns(2)
            with col1:
                # Load original image
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Original PCB Image", use_container_width=True)

                img_array = np.array(image)
                st.markdown("**Image Details**")
                st.write(f"Dimensions: {img_array.shape[1]} x {img_array.shape[0]} pixels")
                st.write(f"File size: {uploaded_file.size / 1024:.2f} KB")

            with col2:
                with st.spinner("Analyzing PCB for defects..."):
                    results = model.predict(img_array, conf=confidence_threshold)

                # Annotated detection result
                annotated = results[0].plot()
                st.image(annotated, caption="Defect Detection Results", use_container_width=True)

                defects_detected = len(results[0].boxes) if results[0].boxes else 0
                st.metric("Defects Detected", defects_detected)

            st.subheader("Detection Details")

            defect_list = []
            if defects_detected > 0:
                with st.expander("View Defect Details", expanded=True):
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Save details for PDF report
                        defect_list.append({
                            "label": label.upper(),
                            "confidence": conf,
                            "location": f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                        })

                        st.markdown(f"""
                        <div class="defect-box">
                            <h4>Defect #{i+1}: {label.upper()}</h4>
                            <p>Confidence: {conf:.2%}</p>
                            <p>Location: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-defect-box">
                    <h4>No Defects Detected</h4>
                    <p>This PCB appears to be within quality standards.</p>
                </div>
                """, unsafe_allow_html=True)

            def generate_pdf_report(annotated_img, defects, file_name="pcb_report.pdf"):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                width, height = A4
                
                # Company branding
                c.setFillColor(colors.HexColor("#0d47a1"))  # Professional blue
                c.rect(0, height - 80, width, 80, fill=True, stroke=False)
                
                c.setFillColor(colors.white)
                c.setFont("Helvetica-Bold", 20)
                c.drawString(50, height - 45, "PRECISION PCB MANUFACTURING")
                
                c.setFont("Helvetica", 12)
                c.drawString(50, height - 65, "Defect Detection Analysis Report")
                
                # Report metadata
                c.setFillColor(colors.black)
                c.setFont("Helvetica", 9)
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.drawString(width - 200, height - 45, f"Report Date: {current_date}")
                c.drawString(width - 200, height - 60, f"Report ID: PCB-{int(time.time())}")
                
                # Separator line
                c.setStrokeColor(colors.HexColor("#0d47a1"))
                c.setLineWidth(1)
                c.line(50, height - 100, width - 50, height - 100)
                
                # Save annotated image temporarily
                annotated_pil = Image.fromarray(annotated_img)
                temp_img_path = "temp_annotated.jpg"
                annotated_pil.save(temp_img_path, quality=95)
                
                # Image with border
                c.setFillColor(colors.white)
                c.setStrokeColor(colors.HexColor("#cccccc"))
                c.setLineWidth(0.5)
                c.rect(50, height - 380, width - 100, 250, fill=True, stroke=True)
                c.drawImage(temp_img_path, 55, height - 375, width=width - 110, height=240, preserveAspectRatio=True)
                
                # Defect summary box
                c.setFillColor(colors.HexColor("#f5f5f5"))
                c.setStrokeColor(colors.HexColor("#e0e0e0"))
                c.rect(50, height - 430, width - 100, 40, fill=True, stroke=True)
                
                c.setFillColor(colors.black)
                c.setFont("Helvetica-Bold", 14)
                if len(defects) == 0:
                    c.drawString(60, height - 415, "QUALITY STATUS: PASS - No Defects Detected")
                    c.setFillColor(colors.HexColor("#388e3c"))  # Green for pass
                else:
                    c.drawString(60, height - 415, f"QUALITY STATUS: FAIL - {len(defects)} Defects Detected")
                    c.setFillColor(colors.HexColor("#d32f2f"))  # Red for fail
                
                c.setFont("Helvetica", 12)
                c.drawString(width - 200, height - 415, f"Defect Count: {len(defects)}")
                
                # Defect details section header
                if defects:
                    c.setFillColor(colors.HexColor("#0d47a1"))
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, height - 470, "DEFECT DETAILS")
                    
                    # Table headers
                    c.setFillColor(colors.HexColor("#e3f2fd"))
                    c.rect(50, height - 490, width - 100, 20, fill=True, stroke=False)
                    c.setFillColor(colors.HexColor("#0d47a1"))
                    c.setFont("Helvetica-Bold", 10)
                    c.drawString(60, height - 485, "ID")
                    c.drawString(100, height - 485, "DEFECT TYPE")
                    c.drawString(250, height - 485, "CONFIDENCE")
                    c.drawString(350, height - 485, "LOCATION")
                    c.drawString(480, height - 485, "STATUS")
                    
                    # List each defect in a table format
                    y = height - 510
                    for i, defect in enumerate(defects, 1):
                        # Alternate row colors
                        if i % 2 == 0:
                            c.setFillColor(colors.HexColor("#f9f9f9"))
                        else:
                            c.setFillColor(colors.white)
                            
                        c.rect(50, y, width - 100, 20, fill=True, stroke=True)
                        
                        c.setFillColor(colors.black)
                        c.setFont("Helvetica", 9)
                        c.drawString(60, y + 5, f"{i}")
                        c.drawString(100, y + 5, f"{defect['label']}")
                        c.drawString(250, y + 5, f"{defect['confidence']:.2%}")
                        c.drawString(350, y + 5, f"{defect['location']}")
                        
                        # Status indicator
                        if defect['confidence'] > 0.8:
                            status = "CRITICAL"
                            color = colors.HexColor("#d32f2f")  # Red
                        elif defect['confidence'] > 0.5:
                            status = "MODERATE"
                            color = colors.HexColor("#f57c00")  # Orange
                        else:
                            status = "MINOR"
                            color = colors.HexColor("#388e3c")  # Green
                            
                        c.setFillColor(color)
                        c.drawString(480, y + 5, status)
                        
                        y -= 25
                        if y < 100:  # new page if overflow
                            c.showPage()
                            # Add header to new page
                            c.setFillColor(colors.HexColor("#0d47a1"))
                            c.setFont("Helvetica-Bold", 12)
                            c.drawString(50, height - 50, "DEFECT DETAILS (CONTINUED)")
                            y = height - 70
                
                # Add footer to each page
                def add_footer(canvas, page_number):
                    canvas.saveState()
                    canvas.setFont('Helvetica', 8)
                    canvas.setFillColor(colors.HexColor("#666666"))
                    canvas.drawString(50, 30, f"Precision PCB Manufacturing - Confidential")
                    canvas.drawCentredString(width / 2, 30, f"Page {page_number}")
                    canvas.drawRightString(width - 50, 30, f"Generated on {current_date}")
                    canvas.restoreState()
                
                # Add footer to first page
                add_footer(c, 1)
                
                # Quality assurance section for the last page
                if len(defects) > 0:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 14)
                    c.setFillColor(colors.HexColor("#0d47a1"))
                    c.drawString(50, height - 50, "QUALITY ASSURANCE RECOMMENDATIONS")
                    
                    c.setFillColor(colors.black)
                    c.setFont("Helvetica", 10)
                    recommendations = [
                        "1. Isolate the defective board from production line",
                        "2. Review manufacturing process at identified defect locations",
                        "3. Check calibration of equipment for potential issues",
                        "4. Document findings in quality management system",
                        "5. Implement corrective actions if similar defects are systematic"
                    ]
                    
                    y = height - 80
                    for rec in recommendations:
                        c.drawString(60, y, rec)
                        y -= 20
                    
                    # Add signature area
                    c.setLineWidth(0.5)
                    c.setStrokeColor(colors.HexColor("#cccccc"))
                    c.line(60, height - 200, 250, height - 200)
                    c.setFont("Helvetica", 9)
                    c.drawString(60, height - 210, "Quality Inspector Signature")
                    
                    c.line(width - 250, height - 200, width - 60, height - 200)
                    c.drawString(width - 250, height - 210, "Quality Manager Signature")
                    
                    # Add footer to this page too
                    add_footer(c, 2)
                
                c.save()
                buffer.seek(0)
                
                # Clean up temporary file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                    
                return buffer
            # Update history
            st.session_state.history.append({
                "filename": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "defects": len(defect_list),
                "details": results[0].boxes  # keep YOLO boxes for later charts
            })

            # Show Download PDF button
            if uploaded_file is not None:
                pdf_buffer = generate_pdf_report(annotated, defect_list)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name="pcb_defect_report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")


# =============================
# TAB 2: BATCH PROCESSING
# =============================
with tab2:
    st.subheader("Batch Processing")
    batch_files = st.file_uploader("Upload multiple PCB images for batch analysis", 
                                   type=["jpg", "jpeg", "png"], 
                                   accept_multiple_files=True,
                                   key="batch_upload")

    if batch_files:
        process_batch = st.button("Process All Images")
        if process_batch:
            progress_bar = st.progress(0)
            results = []

            for i, uploaded_file in enumerate(batch_files):
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)
                result = model.predict(img_array, conf=confidence_threshold)
                annotated = result[0].plot()

                defects = []
                if result[0].boxes:
                    for box in result[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        defects.append({
                            "label": label.upper(),
                            "confidence": conf,
                            "location": f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                        })

                results.append({
                    "filename": uploaded_file.name,
                    "defects": defects,
                    "original": image,
                    "annotated": annotated
                })

                progress_bar.progress((i + 1) / len(batch_files))

            # =============================
            # Summary Display
            # =============================
            st.subheader("Batch Results Summary")
            total_defects = sum([len(r["defects"]) for r in results])
            defective_boards = sum([1 for r in results if len(r["defects"]) > 0])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", len(batch_files))
            col2.metric("Defective Boards", defective_boards)
            col3.metric("Total Defects Found", total_defects)

            st.dataframe(
                data=[{
                    "File": r["filename"],
                    "Defects": len(r["defects"]),
                    "Status": "FAIL" if len(r["defects"]) > 0 else "PASS"
                } for r in results],
                use_container_width=True
            )

            st.subheader("Batch Image Results")
            for r in results:
                st.markdown(f"**{r['filename']}** - Defects: {len(r['defects'])}")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(r["original"], caption="Original", use_container_width=True)
                with col2:
                    st.image(r["annotated"], caption="Defect Detection", use_container_width=True)
                st.divider()

            # =============================
            # PDF Report Generation (Batch)
            # =============================
            def generate_batch_pdf(results, file_name="batch_pcb_report.pdf"):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                width, height = A4
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # ===== Header =====
                c.setFillColor(colors.HexColor("#0d47a1"))
                c.rect(0, height - 80, width, 80, fill=True, stroke=False)
                c.setFillColor(colors.white)
                c.setFont("Helvetica-Bold", 18)
                c.drawString(50, height - 45, "PRECISION PCB MANUFACTURING")
                c.setFont("Helvetica", 11)
                c.drawString(50, height - 65, "Batch Defect Detection Report")

                c.setFillColor(colors.black)
                c.setFont("Helvetica", 9)
                c.drawString(width - 200, height - 45, f"Report Date: {current_date}")
                c.drawString(width - 200, height - 60, f"Total Images: {len(results)}")

                c.line(50, height - 100, width - 50, height - 100)

                # ===== Table Header =====
                y = height - 120
                c.setFillColor(colors.HexColor("#e3f2fd"))
                c.rect(50, y - 5, width - 100, 25, fill=True, stroke=False)
                c.setFillColor(colors.HexColor("#0d47a1"))
                c.setFont("Helvetica-Bold", 8)
                headers = ["ID", "IMAGE", "FILENAME", "DEFECT TYPE(S)", "CONFIDENCE", "LOCATION", "STATUS"]
                x_positions = [55, 80, 150, 250, 350, 420, 520]  # Adjusted for thumbnail column
                for x, header in zip(x_positions, headers):
                    c.drawString(x, y, header)

                # ===== Table Rows =====
                c.setFont("Helvetica", 7)
                y -= 30
                for idx, r in enumerate(results, 1):
                    # Prepare defect info
                    if r["defects"]:
                        defect_types = ", ".join([d["label"] for d in r["defects"]])
                        confidences = ", ".join([f"{d['confidence']:.2%}" for d in r["defects"]])
                        locations = "; ".join([d["location"] for d in r["defects"]])
                        status = "FAIL"
                    else:
                        defect_types = "-"
                        confidences = "-"
                        locations = "-"
                        status = "PASS"

                    # Alternate row color
                    if idx % 2 == 0:
                        c.setFillColor(colors.HexColor("#f9f9f9"))
                        c.rect(50, y - 5, width - 100, 25, fill=True, stroke=False)

                    # Row text
                    c.setFillColor(colors.black)
                    c.drawString(x_positions[0], y, str(idx))
                    c.drawString(x_positions[2], y, r["filename"][:12])   # Shortened filename
                    c.drawString(x_positions[3], y, defect_types[:18])    # Limit width
                    c.drawString(x_positions[4], y, confidences[:12])
                    c.drawString(x_positions[5], y, locations[:18])
                    c.drawString(x_positions[6], y, status)

                    # Add thumbnail (annotated image)
                    thumb_path = f"thumb_{idx}.jpg"
                    annotated_pil = Image.fromarray(r["annotated"])
                    annotated_pil.thumbnail((60, 40))  # Resize for table
                    annotated_pil.save(thumb_path, quality=85)
                    c.drawImage(thumb_path, x_positions[1], y - 5, width=50, height=30, preserveAspectRatio=True)

                    # Remove temporary image
                    if os.path.exists(thumb_path):
                        os.remove(thumb_path)

                    y -= 35
                    if y < 100:  # new page if overflow
                        c.showPage()
                        y = height - 100
                        # Redraw header row on new page
                        c.setFillColor(colors.HexColor("#e3f2fd"))
                        c.rect(50, y - 5, width - 100, 25, fill=True, stroke=False)
                        c.setFillColor(colors.HexColor("#0d47a1"))
                        c.setFont("Helvetica-Bold", 8)
                        for x, header in zip(x_positions, headers):
                            c.drawString(x, y, header)
                        c.setFont("Helvetica", 7)
                        y -= 30

                # ===== Final Summary =====
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.setFillColor(colors.HexColor("#0d47a1"))
                c.drawString(50, height - 50, "BATCH SUMMARY")

                total_defects = sum([len(r["defects"]) for r in results])
                defective_boards = sum([1 for r in results if len(r["defects"]) > 0])

                c.setFont("Helvetica", 11)
                c.setFillColor(colors.black)
                c.drawString(60, height - 90, f"Total Images Processed: {len(results)}")
                c.drawString(60, height - 110, f"Defective Boards: {defective_boards}")
                c.drawString(60, height - 130, f"Total Defects Found: {total_defects}")

                # Recommendations
                if defective_boards > 0:
                    c.setFont("Helvetica-Bold", 12)
                    c.setFillColor(colors.HexColor("#0d47a1"))
                    c.drawString(50, height - 170, "QUALITY RECOMMENDATIONS")
                    c.setFont("Helvetica", 10)
                    c.setFillColor(colors.black)
                    recommendations = [
                        "1. Isolate defective boards for further inspection.",
                        "2. Review defect-prone production steps.",
                        "3. Recalibrate equipment if systematic defects are found.",
                        "4. Document defects in QMS for traceability."
                    ]
                    y = height - 190
                    for rec in recommendations:
                        c.drawString(60, y, rec)
                        y -= 20

                c.save()
                buffer.seek(0)
                return buffer
            # Update history
            st.session_state.history.append({
                "filename": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "defects": len(defect_list),
                "details": results[0].boxes  # keep YOLO boxes for later charts
            })



            pdf_buffer = generate_batch_pdf(results)
            st.download_button(
                label="📥 Download Batch PDF Report",
                data=pdf_buffer,
                file_name="batch_pcb_report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Upload multiple images above to begin batch processing.")

# =============================
# TAB 4: ANALYTICS & HISTORY
# =============================
with tab4:
    st.markdown("### 📊 Analytics & Historical Data")
    
    if not st.session_state.history:
        st.info("No analysis history available. Process some images to see analytics here.")
    else:
        # Convert history to dataframe
        history_df = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_boards = len(history_df)
            st.metric("Total Boards Processed", total_boards)
        
        with col2:
            defective_boards = sum(history_df['defects'] > 0)
            st.metric("Defective Boards", defective_boards)
        
        with col3:
            defect_rate = (defective_boards / total_boards * 100) if total_boards > 0 else 0
            st.metric("Defect Rate", f"{defect_rate:.1f}%")
        
        # Time series of defects
        if 'timestamp' in history_df.columns:
            history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
            daily_defects = history_df.groupby('date')['defects'].sum().reset_index()
            
            fig = px.line(
                daily_defects, 
                x='date', 
                y='defects',
                title="Daily Defects Trend",
                labels={"date": "Date", "defects": "Number of Defects"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Defect type analysis
        st.markdown("#### Defect Type Analysis")
        
        # Extract defect types from history
        defect_types = {}
        for item in st.session_state.history:
            if 'details' in item and item['details']:
                for box in item['details']:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    defect_types[label] = defect_types.get(label, 0) + 1
        
        if defect_types:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(defect_types.values()), 
                    names=list(defect_types.keys()),
                    title="Defect Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=list(defect_types.keys()), 
                    y=list(defect_types.values()),
                    title="Defect Type Count",
                    labels={"x": "Defect Type", "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No defect data available for analysis")
        
        # Recent scans table
        st.markdown("#### Recent Scans")
        recent_df = pd.DataFrame(st.session_state.history)
        if not recent_df.empty:
            # Format for display
            display_df = recent_df[['filename', 'timestamp', 'defects']].copy()
            display_df['status'] = display_df['defects'].apply(lambda x: 'FAIL' if x > 0 else 'PASS')
            display_df = display_df.tail(10)  # Show only last 10 entries
            
            st.dataframe(
                display_df, 
                use_container_width=True,
                column_config={
                    "filename": "File Name",
                    "timestamp": "Timestamp",
                    "defects": "Defects",
                    "status": "Status"
                },
                hide_index=True
            )

# =============================
# TAB 3: REAL-TIME ANALYSIS
# =============================
with tab3:
    st.subheader("Real-Time PCB Analysis")
    st.info("Choose between snapshot or live detection.")

    # ---------------- Snapshot ----------------
    picture = st.camera_input("📸 Take a PCB Snapshot")
    if picture:
        img = Image.open(picture).convert("RGB")
        img_array = np.array(img)
        results = model.predict(img_array, conf=confidence_threshold)
        annotated = results[0].plot()
        st.image(annotated, caption="Detection Result", use_container_width=True)
        defects_detected = len(results[0].boxes) if results[0].boxes else 0
        st.metric("Defects Detected", defects_detected)

    # ---------------- Live Webcam ----------------
    st.markdown("### 🎥 Live Webcam Mode")
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    cam_index = st.selectbox("Select Camera", [0, 1, 2], index=0, help="0 = Laptop, 1/2 = External/iPhone")
    start_webcam = st.button("Start Webcam Analysis")
    stop_webcam = st.button("Stop Webcam")

    if start_webcam:
        st.session_state.run_webcam = True
    if stop_webcam:
        st.session_state.run_webcam = False

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(cam_index)
        stframe = st.empty()

        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Could not access webcam.")
                break

            frame_resized = cv2.resize(frame, (640, 640))
            results = model.predict(frame_resized, conf=confidence_threshold)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        stframe.empty()
        st.success("✅ Webcam stopped.")