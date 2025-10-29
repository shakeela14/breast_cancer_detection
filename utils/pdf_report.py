# utils/pdf_report.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

def create_prediction_pdf(result_text, prob, accuracy, inputs: dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, 750, "Breast Cancer Prediction Report")
    c.setFont("Helvetica", 11)
    c.drawString(60, 720, f"Result: {result_text}")
    c.drawString(60, 700, f"Confidence: {prob:.2%}")
    c.drawString(60, 680, f"Model Accuracy: {accuracy*100:.2f}%")
    c.drawString(60, 650, "Input features:")
    y = 630
    for k, v in inputs.items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 14
        if y < 80:
            c.showPage()
            y = 750
    c.save()
    buffer.seek(0)
    return buffer
