"""
utils.py
--------
Display and analysis helpers for the Transformers Inference Lab.
Task 2: Text Classification utilities.
"""

from typing import List, Dict


# ── ANSI colour helpers (work in terminals & VS Code integrated terminal) ────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _colour_label(label: str) -> str:
    if label.upper() == "POSITIVE":
        return f"{_GREEN}{_BOLD}{label}{_RESET}"
    return f"{_RED}{_BOLD}{label}{_RESET}"


def _confidence_bar(score: float, width: int = 30) -> str:
    """ASCII progress bar representing a confidence score."""
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score * 100:.2f}%"


# ── Main display function ────────────────────────────────────────────────────
def print_results(results: List[Dict]) -> None:
    """
    Pretty-print classification results with confidence bars.

    Parameters
    ----------
    results : List[Dict]
        Output from ``inference.classify_sentences()``.
    """
    separator = f"{_CYAN}{'═' * 72}{_RESET}"
    print(separator)
    print(f"{_BOLD}{_CYAN}  📊  TEXT CLASSIFICATION RESULTS{_RESET}")
    print(separator)

    for idx, item in enumerate(results, 1):
        sentence   = item["sentence"]
        label      = item["label"]
        score      = item["score"]
        all_scores = item["all_scores"]

        print(f"\n  {_BOLD}[{idx}] {sentence}{_RESET}")
        print(f"      Predicted : {_colour_label(label)}")
        print(f"      Confidence: {_confidence_bar(score)}")

        # Show all label scores
        print(f"      {_YELLOW}All label scores:{_RESET}")
        for lbl, sc in all_scores.items():
            marker = " ◀" if lbl == label else "  "
            print(f"        {lbl:<12} {_confidence_bar(sc, width=20)}{marker}")

    print(f"\n{separator}\n")


def print_analysis_summary(results: List[Dict]) -> None:
    """
    Print a brief statistical analysis of the batch results.

    Parameters
    ----------
    results : List[Dict]
        Output from ``inference.classify_sentences()``.
    """
    total     = len(results)
    positives = [r for r in results if r["label"] == "POSITIVE"]
    negatives = [r for r in results if r["label"] == "NEGATIVE"]

    avg_conf      = sum(r["score"] for r in results) / total if total else 0
    avg_pos_conf  = (
        sum(r["score"] for r in positives) / len(positives) if positives else 0
    )
    avg_neg_conf  = (
        sum(r["score"] for r in negatives) / len(negatives) if negatives else 0
    )
    high_conf     = [r for r in results if r["score"] >= 0.99]
    low_conf      = [r for r in results if r["score"] <  0.80]

    separator = f"{_CYAN}{'─' * 72}{_RESET}"
    print(separator)
    print(f"{_BOLD}{_CYAN}  📈  ANALYSIS SUMMARY{_RESET}")
    print(separator)
    print(f"  Total sentences  : {total}")
    print(f"  POSITIVE         : {_GREEN}{_BOLD}{len(positives)}{_RESET}  "
          f"({len(positives)/total*100:.1f}%)")
    print(f"  NEGATIVE         : {_RED}{_BOLD}{len(negatives)}{_RESET}  "
          f"({len(negatives)/total*100:.1f}%)")
    print(f"  Avg confidence   : {avg_conf*100:.2f}%")
    print(f"  Avg POSITIVE conf: {avg_pos_conf*100:.2f}%")
    print(f"  Avg NEGATIVE conf: {avg_neg_conf*100:.2f}%")
    print(f"  High-conf (≥99%) : {len(high_conf)} sentence(s)")
    print(f"  Low-conf  (<80%) : {len(low_conf)}  sentence(s) — worth inspecting!")

    if low_conf:
        print(f"\n  {_YELLOW}⚠  Low-confidence predictions:{_RESET}")
        for r in low_conf:
            print(f"     • [{r['label']} {r['score']*100:.1f}%] {r['sentence'][:70]}")

    print(f"{separator}\n")
