"""Generate synthetic policy PDFs for golden dataset evaluation.

These PDFs contain FULLY SYNTHETIC clinical policy content — no real policies.
Run: python data/golden/policy/generate_pdfs.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_DIR = Path(__file__).parent


def _build_policy_asthma() -> None:
    path = OUTPUT_DIR / "sample_policy_1.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=14)
    subheading = ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=12)
    body = styles["BodyText"]
    elements = []

    elements.append(Paragraph("CLINICAL POLICY BULLETIN: ASTHMA MANAGEMENT", heading))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Policy Number: CPB-2026-ASTH-001 (SYNTHETIC)", body))
    elements.append(Paragraph("Effective Date: January 1, 2026", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("1. Purpose", subheading))
    elements.append(Paragraph(
        "Medical necessity criteria for persistent asthma management in adults (18+), "
        "including step therapy and biologic authorization.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("2. Coverage Criteria — Maintenance Therapy", subheading))
    elements.append(Paragraph(
        "<b>Step 1 (Mild):</b> Low-dose ICS (fluticasone 88-250 mcg/day). No prior auth.<br/>"
        "<b>Step 2 (Moderate):</b> Medium-dose ICS or low-dose ICS + LABA. Requires inadequate control on Step 1 for 4 weeks.<br/>"
        "<b>Step 3 (Severe):</b> High-dose ICS + LABA, consider LAMA. Requires Step 2 adherence for 3 months, ACT &lt; 20.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("3. Biologic Therapy Authorization", subheading))
    elements.append(Paragraph(
        "Covered when ALL met: severe persistent uncontrolled on high-dose ICS + LABA; "
        "documented adherence 6 months; minimum 2 exacerbations requiring systemic corticosteroids in 12 months; "
        "phenotype matching (eos >= 300 for mepolizumab); prior authorization by medical director.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("4. Exclusions", subheading))
    elements.append(Paragraph(
        "Not covered: nebulizers without MDI inability documentation, brand-name when generic available, "
        "compounded meds without documented allergy.", body))

    doc.build(elements)
    print(f"Generated: {path}")


def _build_policy_cardiac_rehab() -> None:
    path = OUTPUT_DIR / "sample_policy_2.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=14)
    subheading = ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=12)
    body = styles["BodyText"]
    elements = []

    elements.append(Paragraph("CLINICAL POLICY BULLETIN: CARDIAC REHABILITATION", heading))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Policy Number: CPB-2026-CARD-002 (SYNTHETIC)", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("1. Covered Indications", subheading))
    elements.append(Paragraph(
        "Medically necessary for: acute myocardial infarction (12 months), coronary artery bypass graft, "
        "PCI/stent, valve repair/replacement, heart failure with EF <= 35%, heart transplant.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("2. Session Limits", subheading))
    elements.append(Paragraph(
        "Up to 36 sessions over 12-week period. Additional 36 (72 total) with documented medical necessity, "
        "functional improvement, and physician attestation.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("3. Exclusions", subheading))
    elements.append(Paragraph(
        "Not covered: maintenance fitness after Phase II, home-based without telemonitoring, "
        "patients with unstable angina or decompensated heart failure.", body))

    doc.build(elements)
    print(f"Generated: {path}")


def _build_policy_joint_replacement() -> None:
    path = OUTPUT_DIR / "sample_policy_3.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=14)
    subheading = ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=12)
    body = styles["BodyText"]
    elements = []

    elements.append(Paragraph("CLINICAL POLICY BULLETIN: TOTAL JOINT REPLACEMENT", heading))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Policy Number: CPB-2026-ORTH-003 (SYNTHETIC)", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("1. Medical Necessity Criteria", subheading))
    elements.append(Paragraph(
        "All required: radiographic evidence (Kellgren-Lawrence 3-4); persistent pain 6+ months; "
        "failure of conservative treatment (physical therapy 6 weeks, NSAIDs, corticosteroid injection); "
        "BMI &lt; 40 (or weight optimization plan if 40-45); no active infection; medical clearance.", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("2. Covered Post-operative Services", subheading))
    elements.append(Paragraph(
        "Physical therapy up to 12 weeks (3x/week tapering), home health nursing (first 2 weeks), "
        "DME (walker then cane), one post-op imaging at 6 weeks. Revision requires separate prior auth.", body))

    doc.build(elements)
    print(f"Generated: {path}")


if __name__ == "__main__":
    _build_policy_asthma()
    _build_policy_cardiac_rehab()
    _build_policy_joint_replacement()
    print(f"\nAll policy golden PDFs generated in: {OUTPUT_DIR}")
