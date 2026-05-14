"""Generate synthetic ADR PDFs for golden dataset evaluation.

These PDFs contain FULLY SYNTHETIC data — no real PHI.
Run: python data/golden/adr/generate_pdfs.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_DIR = Path(__file__).parent


def _build_adr_emergency_visit() -> None:
    path = OUTPUT_DIR / "sample_adr_1.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=14)
    subheading = ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=12)
    body = styles["BodyText"]
    elements = []

    elements.append(Paragraph("EMERGENCY DEPARTMENT VISIT NOTE", heading))
    elements.append(Spacer(1, 12))
    demographics = [
        ["Patient Name:", "Jane A. Doe (SYNTHETIC)"], ["DOB:", "1985-03-15"],
        ["MRN:", "SYN-000111"], ["Visit Date:", "2026-01-10"], ["Provider:", "Dr. Michael Chen, MD"],
    ]
    t = Table(demographics, colWidths=[2 * inch, 4 * inch])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Chief Complaint", subheading))
    elements.append(Paragraph(
        "45-year-old female presenting with acute shortness of breath, wheezing, and chest tightness "
        "for the past 6 hours. Third emergency visit this year for similar symptoms.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("History of Present Illness", subheading))
    elements.append(Paragraph(
        "Known history of moderate persistent asthma diagnosed in 2018. Ran out of maintenance inhaler "
        "(fluticasone/salmeterol) two weeks ago. Symptoms worsened after cold air exposure. "
        "Used albuterol rescue inhaler 4 times today without relief.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Primary Diagnosis", subheading))
    elements.append(Paragraph(
        "<b>Acute asthma exacerbation</b> (ICD-10: J45.41) — moderate severity. "
        "Contributing factor: medication non-adherence.", body))
    elements.append(Spacer(1, 200))

    elements.append(Paragraph("Treatment Administered", subheading))
    elements.append(Paragraph(
        "1. Nebulized albuterol 2.5 mg x 3<br/>2. Ipratropium bromide 0.5 mg x 1<br/>"
        "3. Prednisone 40 mg PO x 1<br/>4. Supplemental oxygen 2L (SpO2 91% to 97%)", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Disposition", subheading))
    elements.append(Paragraph(
        "Discharged home stable. Prescribed prednisone 40 mg x 5 days, albuterol PRN, "
        "fluticasone/salmeterol resumed. Follow-up PCP 7 days. Pulmonology referral.", body))

    doc.build(elements)
    print(f"Generated: {path}")


def _build_adr_discharge_summary() -> None:
    path = OUTPUT_DIR / "sample_adr_2.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=14)
    subheading = ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=12)
    body = styles["BodyText"]
    elements = []

    elements.append(Paragraph("HOSPITAL DISCHARGE SUMMARY", heading))
    elements.append(Spacer(1, 12))
    demographics = [
        ["Patient Name:", "Robert J. Smith (SYNTHETIC)"], ["DOB:", "1958-11-22"],
        ["MRN:", "SYN-000222"], ["Admission Date:", "2026-01-05"],
        ["Discharge Date:", "2026-01-09"], ["Attending:", "Dr. Sarah Williams, MD"],
    ]
    t = Table(demographics, colWidths=[2 * inch, 4 * inch])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Admission Diagnosis", subheading))
    elements.append(Paragraph(
        "<b>Acute decompensated heart failure</b> (ICD-10: I50.31) with volume overload. "
        "Secondary: Type 2 diabetes, hypertension, CKD stage 3.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Hospital Course", subheading))
    elements.append(Paragraph(
        "67-year-old male with 3-day progressive dyspnea, bilateral edema, 8-pound weight gain. "
        "BNP 1,850. CXR: bilateral effusions. EF on echo: 30% (reduced from 40%). "
        "IV furosemide 40 mg BID, transitioned to oral day 3. Weight decreased 6 kg.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Discharge Medications", subheading))
    elements.append(Paragraph(
        "1. Furosemide 40 mg BID<br/>2. Lisinopril 20 mg daily<br/>"
        "3. Carvedilol 12.5 mg BID<br/>4. Metformin 1000 mg BID<br/>"
        "5. Aspirin 81 mg daily<br/>6. KCl 20 mEq daily", body))

    doc.build(elements)
    print(f"Generated: {path}")


def _build_adr_surgical_note() -> None:
    path = OUTPUT_DIR / "sample_adr_3.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=14)
    subheading = ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=12)
    body = styles["BodyText"]
    elements = []

    elements.append(Paragraph("POST-OPERATIVE FOLLOW-UP NOTE", heading))
    elements.append(Spacer(1, 12))
    demographics = [
        ["Patient Name:", "Maria L. Garcia (SYNTHETIC)"], ["DOB:", "1970-07-08"],
        ["MRN:", "SYN-000333"], ["Visit Date:", "2026-01-15"],
        ["Surgeon:", "Dr. James Park, MD"], ["Procedure Date:", "2025-12-20"],
    ]
    t = Table(demographics, colWidths=[2 * inch, 4 * inch])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Procedure Performed", subheading))
    elements.append(Paragraph(
        "<b>Right total knee arthroplasty</b> (CPT: 27447) for severe osteoarthritis (ICD-10: M17.11).", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Current Status (4 weeks post-op)", subheading))
    elements.append(Paragraph(
        "Pain 3/10 at rest, 5/10 with PT. ROM: 5-95 degrees (goal 0-120 by 3 months). "
        "Incision well-healed. Weight-bearing as tolerated with walker. PT 3x/week progressing well.", body))

    doc.build(elements)
    print(f"Generated: {path}")


if __name__ == "__main__":
    _build_adr_emergency_visit()
    _build_adr_discharge_summary()
    _build_adr_surgical_note()
    print(f"\nAll ADR golden PDFs generated in: {OUTPUT_DIR}")
