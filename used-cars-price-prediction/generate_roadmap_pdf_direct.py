from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def create_roadmap_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=colors.grey
    )
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=18,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.darkblue,
        borderPadding=5,
        borderColor=colors.lightgrey,
        borderWidth=1
    )
    
    sub_section_style = ParagraphStyle(
        'SubSection',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.black
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=20,
        bulletIndent=10
    )

    story = []

    # Title Page
    story.append(Paragraph("Frontend to AI Engineer Roadmap", title_style))
    story.append(Paragraph("2026 Edition • 6-9 Month Plan • 30h/Week", subtitle_style))
    story.append(Spacer(1, 20))

    # Overview Table
    data = [
        ['Phase', 'Focus', 'Duration', 'Key Outcome'],
        ['1', 'Deep Learning Foundations', 'Months 1-2', '3-4 DL Projects'],
        ['2', 'MLOps & Production', 'Months 3-4', 'Cloud Deployment'],
        ['3', 'Full-Stack Portfolio', 'Months 5-6', '2 Killer Projects'],
        ['4', 'Job Hunt & Prep', 'Months 7-9', 'Offers']
    ]
    
    t = Table(data, colWidths=[40, 160, 80, 140])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 30))

    # Phase 1
    story.append(Paragraph("Phase 1: Deep Learning Foundations (Months 1-2)", section_style))
    story.append(Paragraph("Goal: Master PyTorch, Computer Vision, and NLP.", body_style))
    
    phases = [
        ("Week 1-2: PyTorch Basics", [
            "Study (15h): PyTorch '60 Minute Blitz', Tensors, Backprop.",
            "Code (15h): Re-implement MNIST, Feed-Forward NN from scratch."
        ]),
        ("Week 3-4: Computer Vision", [
            "Study (15h): CNNs (ResNet), Object Detection (YOLO).",
            "Project: 'Real-time Face Mask Detector'."
        ]),
        ("Week 5-6: NLP", [
            "Study (15h): Transformers, BERT, GPT basics.",
            "Project: Sentiment Analysis & Simple RAG bot."
        ]),
        ("Week 7-8: Generative AI", [
            "Study (15h): Diffusion Models, LLMs.",
            "Project: AI Art Generator w/ Stable Diffusion API."
        ])
    ]

    for title, points in phases:
        story.append(Paragraph(title, sub_section_style))
        for p in points:
            story.append(Paragraph(f"• {p}", bullet_style))

    # Phase 2
    story.append(Paragraph("Phase 2: MLOps & Production (Months 3-4)", section_style))
    story.append(Paragraph("Goal: Deploy models professionally.", body_style))
    
    phases_2 = [
        ("Week 9-10: Serving", [
            "Backend: FastAPI + Docker.",
            "Optimization: ONNX export, Quantization."
        ]),
        ("Week 11-12: Cloud", [
            "Deploy to AWS Lambda or Google Cloud Run.",
            "Store artifacts in S3/GCS."
        ]),
        ("Week 13-14: Pipelines", [
            "Orchestration: Airflow/Prefect.",
            "Tracking: MLflow or Weights & Biases."
        ]),
        ("Week 15-16: Edge AI (Your Edge!)", [
            "TensorFlow.js or ONNX Runtime Web.",
            "Project: In-browser Object Detection."
        ])
    ]

    for title, points in phases_2:
        story.append(Paragraph(title, sub_section_style))
        for p in points:
            story.append(Paragraph(f"• {p}", bullet_style))

    # Phase 3
    story.append(Paragraph("Phase 3: The Killer Portfolio (Months 5-6)", section_style))
    
    projects = [
        ("Project A: The SaaS (Weeks 17-20)", [
            "Idea: AI-Powered Document Assistant.",
            "Tech: Next.js + FastAPI + LangChain + Vector DB.",
            "Outcome: Full-stack RAG application."
        ]),
        ("Project B: Deep Tech (Weeks 21-24)", [
            "Idea: Industrial Safety Object Detection.",
            "Tech: YOLOv8 custom training + Streamlit Dashboard.",
            "Deployment: AWS EC2 with GPU."
        ])
    ]

    for title, points in projects:
        story.append(Paragraph(title, sub_section_style))
        for p in points:
            story.append(Paragraph(f"• {p}", bullet_style))

    # Weekly Schedule
    story.append(Paragraph("Weekly Schedule Strategy", section_style))
    
    schedule_data = [
        ['Day', 'Hours', 'Focus'],
        ['Mon', '3.5h', 'Study Theory'],
        ['Tue', '3.5h', 'Hands-on Code'],
        ['Wed', '3.5h', 'Hands-on Code'],
        ['Thu', '3.5h', 'Project Work'],
        ['Fri', '3.5h', 'Project Work'],
        ['Sat', '6.5h', 'Deep Dive'],
        ['Sun', '6.5h', 'Review + Prep']
    ]

    t2 = Table(schedule_data, colWidths=[60, 60, 200])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t2)

    doc.build(story)
    print(f"PDF successfully generated: {filename}")

if __name__ == "__main__":
    create_roadmap_pdf('AI_Engineer_Roadmap.pdf')
