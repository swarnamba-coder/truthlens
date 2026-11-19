"""Court-ready PDF report generation using ReportLab, ensuring complete evidence traceability, candidate model governance records, and non-combined raw findings."""
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from app.config import REPORTS, TOOL_REGISTRY
from app.database.database import Investigation, EvidenceFile, AgentExecution, Finding, iso

def build(db, inv_id: str):
    inv = db.get(Investigation, inv_id)
    files = db.query(EvidenceFile).filter_by(investigation_id=inv_id).all()
    execs = {e.name: e for e in db.query(AgentExecution).filter_by(investigation_id=inv_id).all()}
    findings = [f.data for f in db.query(Finding).filter_by(investigation_id=inv_id).all()]

    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name='MetaKey', parent=ss['BodyText'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#1e293b')))
    ss.add(ParagraphStyle(name='MetaVal', parent=ss['BodyText'], fontName='Helvetica', fontSize=9, textColor=colors.HexColor('#475569')))
    ss.add(ParagraphStyle(name='SmallMono', parent=ss['BodyText'], fontName='Courier', fontSize=7.5, leading=9, textColor=colors.HexColor('#0f172a')))

    P = lambda t, s="BodyText": Paragraph(escape(str(t)), ss[s])

    story = [
        P("TruthLens Forensic Intelligence Report", "Title"),
        P(f"Case ID: {inv.id} — {inv.title}", "Heading3"),
        P(f"Generated: {iso(inv.created_at)}" + (f" | Case Reference: {inv.case_ref}" if inv.case_ref else "")),
        Spacer(1, 6),
        P("CRITICAL NOTICE: Automated indicators support human investigation. They do NOT independently establish authenticity or prove intentional manipulation. Findings must be evaluated by a qualified examiner.", "Italic"),
        Spacer(1, 10),
        P("Investigation Overview", "Heading2"),
        P(inv.description or "No case description provided."),
        Spacer(1, 10),
        P("Evidence Inventory & Cryptographic Traceability", "Heading2")
    ]

    # Evidence files table with SHA-256 hashes
    headers = ["File Name", "Modality", "Size (bytes)", "SHA-256 Digest"]
    rows = [[Paragraph(h, ss["MetaKey"]) for h in headers]]
    for f in files:
        rows.append([
            Paragraph(escape(f.original_name), ss["BodyText"]),
            Paragraph(f.kind.upper(), ss["BodyText"]),
            Paragraph(f"{f.size:,}", ss["BodyText"]),
            Paragraph(f.sha256, ss["SmallMono"])
        ])
    t_ev = Table(rows, colWidths=[130, 60, 70, 240], repeatRows=1)
    t_ev.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t_ev, Spacer(1, 6), P("Cryptographic SHA-256 hashes confirm file integrity from the moment of ingest. A hash proves identity; it does not prove origin.", "Italic"), Spacer(1, 10)]

    # Agent execution states
    story.append(P("Agent Execution & Methodology Status", "Heading2"))
    exec_rows = [["Agent Pipeline", "Execution State", "Execution Notes"]]
    for name, e in execs.items():
        exec_rows.append([name.upper(), e.state.upper(), e.reason or "Executed without error"])
    t_ex = Table([[Paragraph(escape(str(c)), ss["BodyText"]) for c in r] for r in exec_rows], colWidths=[120, 100, 280], repeatRows=1)
    t_ex.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
    ]))
    story += [t_ex, Spacer(1, 12)]

    # Findings by Modality
    modalities = [
        ("image", "Image Forensics Findings (Metadata, ELA, C2PA)"),
        ("video", "Video Forensics Findings (Stream Parameters, Temporal Anomalies)"),
        ("audio", "Audio Forensics Findings (DSP Features, Spectrogram, Whisper)"),
        ("text", "Text Context Findings (Difflib Transcript Alignment)"),
        ("provenance", "Provenance & Content Credentials Findings (C2PA)"),
        ("correlation", "Cross-Modal Correlation & Temporal Clustering")
    ]

    for mod, title in modalities:
        fs = [f for f in findings if f.get("modality") == mod]
        if not fs:
            continue
        story.append(P(title, "Heading2"))
        for f in fs:
            ts = f" at {f['timestamp_seconds']}s" if f.get("timestamp_seconds") is not None else ""
            prov = f.get("provenance", "observed").upper()
            story.append(P(f"[{prov}] {f['title']}{ts}", "Heading4"))
            story.append(P(f.get("description", "")))
            det = f.get("detector") or {}
            if det:
                story.append(P(f"Detector / Model: {det.get('name', 'N/A')} (v{det.get('version', '1')}) | License: {det.get('license', 'N/A')} | Mode: {det.get('execution_mode', 'N/A')}", "MetaVal"))
            if f.get("score") is not None:
                story.append(P(f"Raw detector metric: {f['score']} (Note: raw detector metric, NOT an authenticity probability)", "MetaKey"))
            for alt in f.get("alternative_explanations", []):
                story.append(P(f"Alternative Benign Explanation: {alt}", "Italic"))
            for lim in f.get("limitations", []):
                story.append(P(f"Forensic Limitation: {lim}", "Italic"))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))

    # Candidate Models & Tool Governance Registry
    story.append(PageBreak())
    story.append(P("Candidate Model & Repository Governance Registry", "Heading2"))
    story.append(P("TruthLens evaluates candidate open-source models across 8 criteria (license, hardware, pretrained weights, turnaround, media compatibility, version, source, and transparent fallback). Below is the active evaluation status:", "BodyText"))
    story.append(Spacer(1, 6))

    reg_rows = [["Tool / Model", "Category", "License", "Hardware", "Integration Status"]]
    for m in TOOL_REGISTRY[:10]:
        reg_rows.append([
            m["name"],
            m["category"],
            m["license"],
            m["hardware_requirements"],
            m["integration_status"]
        ])
    t_reg = Table([[Paragraph(escape(str(c)), ss["SmallMono"]) for c in r] for r in reg_rows], colWidths=[110, 70, 90, 110, 120], repeatRows=1)
    t_reg.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [t_reg, Spacer(1, 14)]

    # Final recommendations
    story.append(P("Examiner Review Recommendations", "Heading2"))
    story.append(P("1. Verification of Chain-of-Custody: Ensure original media acquisition records and camera firmware logs are matched against extracted EXIF/container headers."))
    story.append(P("2. Alternative Cause Analysis: Verify whether video re-encoding, transmission packet loss, social media compression, or room acoustic echo could explain the flagged timestamps."))
    story.append(P("3. Multi-Model Corroboration: Never rely on a single heuristic or classifier output."))
    story.append(Spacer(1, 10))
    story.append(P("TruthLens — Digital Truth. Forensic Intelligence.", "Italic"))

    path = REPORTS / f"{inv_id}.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, title=f"TruthLens {inv_id}").build(story)
    return path
