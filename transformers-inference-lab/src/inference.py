"""
inference.py
------------
Core inference logic for the Transformers Inference Lab.
Task 2: Text Classification using distilbert-base-uncased-finetuned-sst-2-english
"""

from transformers import pipeline
import torch


def build_classification_pipeline(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: int = -1,          # -1 = CPU, 0 = first GPU
) -> pipeline:
    """
    Instantiate a HuggingFace text-classification pipeline.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub id.
    device : int
        Device index. -1 means CPU.

    Returns
    -------
    transformers.Pipeline
        Ready-to-use text classification pipeline.
    """
    print(f"\n🔧  Loading pipeline: '{model_name}' ...")
    clf = pipeline(
        task="text-classification",
        model=model_name,
        device=device,
        top_k=None,               # get scores for ALL labels, not just top-1 (replaces deprecated return_all_scores=True)
    )
    print("✅  Pipeline loaded successfully!\n")
    return clf


def classify_sentences(clf: pipeline, sentences: list[str]) -> list[dict]:
    """
    Run inference on a list of sentences and return structured results.

    Parameters
    ----------
    clf : pipeline
        A loaded text-classification pipeline.
    sentences : list[str]
        Input texts to classify.

    Returns
    -------
    list[dict]
        Each item contains:
          - sentence   : original input text
          - label      : predicted class (POSITIVE / NEGATIVE)
          - score      : confidence of the predicted class (0-1)
          - all_scores : dict mapping every label to its confidence
    """
    raw_outputs = clf(sentences)   # list of list[dict] when return_all_scores=True

    results = []
    for sentence, label_scores in zip(sentences, raw_outputs):
        # label_scores is a list like:
        # [{"label": "NEGATIVE", "score": 0.002}, {"label": "POSITIVE", "score": 0.998}]
        scores_dict = {ls["label"]: ls["score"] for ls in label_scores}
        top = max(label_scores, key=lambda x: x["score"])
        results.append(
            {
                "sentence": sentence,
                "label": top["label"],
                "score": top["score"],
                "all_scores": scores_dict,
            }
        )

    return results
