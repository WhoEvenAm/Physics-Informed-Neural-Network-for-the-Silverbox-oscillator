"""Build the written Task 5 section from the saved experiment results.

Requires ReportLab: pip install -r requirements-report.txt
Run: python src/build_report.py --repository-url https://github.com/USER/REPO
Without a URL, the document is explicitly marked as a draft.
"""

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
)


PROJECT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url")
    args = parser.parse_args()
    best = json.loads((PROJECT / "output" / "best_result.json").read_text())
    rows = json.loads((PROJECT / best["results_directory"] / "results.json").read_text())
    predictions = np.loadtxt(PROJECT / "output" / "best_predictions.csv", delimiter=",", skiprows=1)
    error = predictions[3072:, 2] - predictions[3072:, 1]
    checkpoint_config = {
        key: best[key] for key in ("optimizer", "learning_rate", "steps", "rollout_length", "batch_size", "seed")
    }
    width = A4[0] - 88
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportBody", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=23, leading=28, spaceAfter=12))
    styles.add(ParagraphStyle("ReportHeading", fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle("Equation", fontName="Courier", fontSize=11, leading=17, alignment=TA_CENTER, spaceAfter=9))
    styles.add(ParagraphStyle("SmallText", fontName="Helvetica", fontSize=8.5, leading=11, spaceAfter=6))
    story = []

    def paragraph(text, style="ReportBody"):
        story.append(Paragraph(text, styles[style]))

    def table(data, widths=None):
        item = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#19374a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEADING", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef4f7"), colors.white]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#19374a")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story.append(item)
        story.append(Spacer(1, 9))

    title = "Task 5: Silverbox oscillator" + (" (draft)" if not args.repository_url else "")
    paragraph(title, "ReportTitle")
    paragraph("Neural acceleration modeling with a continuous RK4 free run", "ReportHeading")
    paragraph(
        "A fully connected PyTorch network predicts acceleration from position, velocity, "
        "and applied voltage. Integrating its state equation produces the output waveform. "
        "Three architectures and three training settings per architecture were compared, "
        "alongside the original tutorial configuration."
    )
    paragraph("Selected result", "ReportHeading")
    table([
        ["Measurement", "Value"],
        ["Hidden-layer widths", best["architecture"]],
        ["Trainable parameters", str(best["parameters"])],
        ["Full free-run training RMSE", f"{best['train_rmse']:.8f} V"],
        ["Full free-run test RMSE", f"{best['test_rmse']:.8f} V"],
        ["Zero-prediction test RMSE", f"{best['zero_test_rmse']:.8f} V"],
    ], [width * 0.65, width * 0.35])
    paragraph("Data preparation", "ReportHeading")
    paragraph(
        "The source is SNLS80mV.mat: channel V1 is input voltage u and channel V2 is output "
        "voltage y. The slice [42650:46746] contains 4,096 samples at 610.35 Hz. As required, "
        "the mean of each signal on the full excerpt is subtracted. The first 3,072 samples "
        "form the training record; the last 1,024 form the test record."
    )
    paragraph("Physical model and state equation", "ReportHeading")
    paragraph("y'' + a y' + b y + c y<super>3</super> = g u(t)", "Equation")
    paragraph("y' = v;    v' = alpha_theta(y, v, u)", "Equation")
    paragraph(
        "The Duffing equation motivates the acceleration model. The network represents "
        "the force; a, b, c, and g are not fitted as separate coefficients here. The state "
        "is [y, v]. Classical RK4 advances it with dt = 1/610.35 s, linearly interpolating "
        "the known input voltage between adjacent samples."
    )

    story.append(PageBreak())
    paragraph("Training and model comparison", "ReportTitle")
    paragraph(
        "Every network has three inputs ordered [y, v, u], Tanh after each hidden linear "
        "layer, and one linear acceleration output. Training-only RMS scales are approximately "
        "0.05302 V for position, 22.1815 V/s for finite-difference velocity, 0.02196 V for input, "
        "and 9,886.33 V/s^2 for acceleration. Physical states remain unscaled inside RK4."
    )
    paragraph(
        "Each update samples windows entirely within training data. Starting position is "
        "measured training y; starting velocity is (y[1]-y[0])/dt within that window. Later "
        "states are predicted, without measured-position corrections. The loss is mean "
        "squared position error averaged across time and batch. Backpropagation passes "
        "through RK4. Weight initialization uses seed 0; window sampling uses a separate "
        "generator with seed 1 for comparisons independent of network width. The original "
        "tutorial run also appears in the table; it uses the global sampling sequence "
        "after seed-0 weight initialization."
    )
    paragraph("Training settings tested for every architecture", "ReportHeading")
    settings = []
    for number in (1, 2, 3):
        row = next(row for row in rows if row["name"].endswith(f"setting{number}"))
        settings.append([
            str(number), row["optimizer"], str(row["learning_rate"]),
            str(row["steps"]), str(row["rollout_length"]), str(row["batch_size"]),
        ])
    table([["Setting", "Optimizer", "Rate", "Updates", "Window", "Batch"]] + settings,
          [50, 85, 75, 90, 100, width - 400])
    paragraph("AdamW uses weight decay 0.0001. Settings vary several factors together; this is a small search rather than an isolated causal study.", "SmallText")
    comparison = [["Architecture", "Setting", "Params", "Train mV", "Test mV", "Time (s)"]]
    for row in rows:
        complete = row["status"] == "complete"
        comparison.append([
            row["architecture"], row["name"].rsplit("setting", 1)[1],
            str(row.get("parameters", "-")),
            f"{1000 * row['train_rmse']:.3f}" if complete else "Failed",
            f"{1000 * row['test_rmse']:.3f}" if complete else "Failed",
            f"{row['training_seconds']:.1f}" if complete else "-",
        ])
    table(comparison, [105, 50, 70, 100, 100, width - 425])
    paragraph(
        f"The selected version is <b>{escape(best['name'])}</b>: "
        f"{best['optimizer']}, learning rate {best['learning_rate']}, {best['steps']} updates, "
        f"{best['rollout_length']}-sample windows, and batch size {best['batch_size']}. "
        "It gives the lowest test free-run RMSE in this comparison, as required by the "
        "assignment. Test outputs enter scoring and model selection after fitting; "
        "they never enter the gradient calculation."
    )

    story.append(PageBreak())
    paragraph("Continuous test prediction", "ReportTitle")
    paragraph(
        "After weights are fixed, the model starts at excerpt sample 0 with measured position "
        "and forward finite-difference velocity. It integrates all 4,096 samples in one run. "
        "The state is carried across the training/test boundary. Measured test y is used "
        "only to calculate error and draw this comparison after prediction."
    )
    story.append(Image(str(PROJECT / "output" / "best_test_predictions.png"), width=width, height=width * 0.75))
    improvement = 100 * (1 - best["test_rmse"] / best["zero_test_rmse"])
    paragraph(
        f"Test RMSE is {1000 * best['test_rmse']:.3f} mV, a {improvement:.1f}% reduction "
        f"from the zero-prediction baseline. Maximum absolute test error is "
        f"{1000 * np.max(np.abs(error)):.3f} mV; mean signed error is "
        f"{1000 * np.mean(error):.3f} mV. The lower panel displays individual signed errors; "
        "RMSE summarizes their squared magnitudes."
    )
    paragraph("Code and reproducibility", "ReportHeading")
    paragraph("Run the comparison: <font name='Courier'>python src/experiments.py</font><br/>"
              "Evaluate saved weights: <font name='Courier'>python src/test.py --no-show</font>", "SmallText")
    paragraph(
        "Saved artifacts include the experiment CSV/JSON tables, per-run loss histories "
        "and checkpoints, selected weights, and full measured/predicted traces. The "
        "selected checkpoint was reloaded and its RMSE reproduced independently."
    )
    if args.repository_url:
        url = escape(args.repository_url, quote=True)
        paragraph(f"Repository: <link href='{url}' color='#2563eb'>{url}</link>", "SmallText")
    else:
        paragraph("Repository URL: pending publication. This draft needs the repository link before submission.", "SmallText")
    paragraph("Source: Homework 1, Task 5, pages 5-6; Silverbox SNLS80mV.mat benchmark record (nonlinearbenchmark.org).", "SmallText")

    output = PROJECT / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    document_path = output / "task5_report.pdf"

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536675"))
        canvas.drawString(44, 24, "Task 5 | Silverbox neural acceleration model")
        canvas.drawRightString(A4[0] - 44, 24, str(document.page))

    SimpleDocTemplate(
        str(document_path), pagesize=A4, leftMargin=44, rightMargin=44,
        topMargin=38, bottomMargin=42, title=title,
    ).build(story, onFirstPage=footer, onLaterPages=footer)

    markdown = [
        f"# {title}", "", "## Selected model", "",
        f"- Hidden widths: {best['architecture']}; parameters: {best['parameters']}",
        f"- Settings: {checkpoint_config}",
        f"- Training free-run RMSE: {best['train_rmse']:.8f} V",
        f"- Test free-run RMSE: {best['test_rmse']:.8f} V", "",
        "## Experiment comparison", "",
        "| Architecture | Setting | Train RMSE (V) | Test RMSE (V) |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "complete":
            markdown.append(f"| {row['architecture']} | {row['name'].rsplit('setting', 1)[1]} | {row['train_rmse']:.8f} | {row['test_rmse']:.8f} |")
    markdown.extend(["", "Repository: " + (args.repository_url or "Pending publication"), ""])
    (PROJECT / "output" / "task5_summary.md").write_text("\n".join(markdown))
    print("Report saved:", document_path)


if __name__ == "__main__":
    main()
