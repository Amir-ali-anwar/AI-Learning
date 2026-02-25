"""
task2_text_classification.py
────────────────────────────
Task 2: Text Classification (Without Fine-Tuning)
Model : distilbert-base-uncased-finetuned-sst-2-english

Run:
    python task2_text_classification.py

Goals:
  ✅  Feed custom sentences to a pre-trained pipeline
  ✅  Observe POSITIVE / NEGATIVE predictions
  ✅  Analyse confidence scores for every label
  ✅  Understand the inference pipeline end-to-end
"""

import sys
import os

# Make sure the src package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from inference import build_classification_pipeline, classify_sentences
from utils    import print_results, print_analysis_summary


# ── 1.  Custom sentences ─────────────────────────────────────────────────────
#
#   These sentences are intentionally diverse:
#     • Clear positive / negative cases  → should show high confidence
#     • Borderline / ambiguous cases     → should show lower confidence
#     • Negation traps                   → tests the model's linguistic depth
#     • Domain-specific phrases          → probes generalisation
# ─────────────────────────────────────────────────────────────────────────────
SENTENCES = [
    # ── Clear positives ──────────────────────────────────────────────────────
    "This movie is absolutely fantastic — I loved every second of it!",
    "The customer service was outstanding and the staff were incredibly kind.",
    "I have never tasted such an amazing meal in my entire life.",

    # ── Clear negatives ──────────────────────────────────────────────────────
    "This product is a complete waste of money and I deeply regret buying it.",
    "Terrible experience. The room was dirty, cold, and the staff were rude.",
    "I hated this book. The plot was boring and the characters were flat.",

    # ── Borderline / ambiguous ───────────────────────────────────────────────
    "The film was okay — not great, but not terrible either.",
    "It is what it is. Some parts were decent, others felt forced.",

    # ── Negation traps ───────────────────────────────────────────────────────
    "Not bad at all! Actually quite impressive for the price.",
    "I wouldn't say this was a bad experience, but it definitely wasn't memorable.",

    # ── Mixed sentiment ──────────────────────────────────────────────────────
    "The food was delicious but the service was appallingly slow.",
    "Beautiful location, however the hotel rooms left much to be desired.",

    # ── Domain-specific ──────────────────────────────────────────────────────
    "The model achieved state-of-the-art results on all benchmarks.",
    "Our quarterly losses increased dramatically despite best efforts.",
]


# ── 2.  Main inference pipeline ──────────────────────────────────────────────
def main() -> None:
    print("\n" + "═" * 72)
    print("  🤗  TASK 2 — TEXT CLASSIFICATION (No Fine-Tuning)")
    print("  Model: distilbert-base-uncased-finetuned-sst-2-english")
    print("═" * 72)

    # Step A: Build the pipeline (downloads model weights on first run)
    clf = build_classification_pipeline()

    # Step B: Run inference on all sentences
    print(f"🚀  Running inference on {len(SENTENCES)} sentences ...\n")
    results = classify_sentences(clf, SENTENCES)

    # Step C: Display detailed per-sentence results
    print_results(results)

    # Step D: Print aggregate analysis
    print_analysis_summary(results)

    # Step E: Educational observations
    _print_observations()


def _print_observations() -> None:
    """Print learning-oriented observations about the inference pipeline."""
    print("─" * 72)
    print("  📚  KEY OBSERVATIONS")
    print("─" * 72)
    observations = [
        ("Inference Pipeline",
         "We used HuggingFace `pipeline('text-classification')`. It chains\n"
         "       tokenisation → model forward-pass → softmax → label decoding\n"
         "       in a single call — no fine-tuning needed."),

        ("Model Architecture",
         "DistilBERT is a distilled (smaller, faster) version of BERT.\n"
         "       It keeps 97% of BERT's language understanding at 60% of the size\n"
         "       and 2× the inference speed."),

        ("Confidence Scores",
         "Scores are softmax probabilities summing to 1.0 across labels.\n"
         "       A score of 0.99 means the model is 99% confident.\n"
         "       Scores near 0.5 indicate the model is uncertain."),

        ("High Confidence",
         "Strongly opinionated sentences (no ambiguity, clear vocabulary)\n"
         "       tend to receive confidence scores > 0.99."),

        ("Low Confidence / Ambiguity",
         "Mixed-sentiment or borderline sentences score near 0.5–0.80,\n"
         "       revealing the model's uncertainty — useful for flagging edge cases."),

        ("Negation Handling",
         "DistilBERT handles negation reasonably well (e.g., 'not bad')\n"
         "       thanks to its contextual attention mechanism."),

        ("No Fine-Tuning",
         "All predictions come purely from the pre-trained SST-2 checkpoint.\n"
         "       Fine-tuning on domain-specific data would improve accuracy."),
    ]
    for title, body in observations:
        print(f"\n  🔹 {title}")
        print(f"       {body}")
    print("\n" + "─" * 72 + "\n")


if __name__ == "__main__":
    main()
