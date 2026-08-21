# metrics.py — accuracy scoring for short ground-truth answers.
#
# Metric choice (justification in the report):
#   Text-in-the-wild QA has short factual answers, and the model answers in
#   full sentences. We therefore use **answer recall** as the headline: the
#   fraction of ground-truth answer tokens that appear in the prediction
#   (1.0 = the answer is fully contained). This is what a voice assistant
#   needs — it doesn't matter if the model says "The price is $5.50." as long
#   as the answer "5.50" is in there. We also report strict Exact Match (EM)
#   and SQuAD-style word-F1 as secondary checks.
#
# Normalization: lowercase, strip currency symbols and sentence punctuation,
# collapse whitespace, and collapse "7 am" -> "7am" so time formats match.

import re

CURRENCY_RE = re.compile(r"[\$€£]")
SPACE_RE = re.compile(r"\s+")
NONALNUM_RE = re.compile(r"[^a-z0-9 .%-]")
TRAILING_DOT_RE = re.compile(r"\.(?=\s*$)")
TIME_SPACE_RE = re.compile(r"(\d)\s+?(am|pm)")  # "7 am" -> "7am"


def normalize(s):
    s = str(s).lower()
    s = CURRENCY_RE.sub("", s)
    s = NONALNUM_RE.sub(" ", s)
    s = TRAILING_DOT_RE.sub("", s)
    s = SPACE_RE.sub(" ", s)
    s = TIME_SPACE_RE.sub(r"\1\2", s)
    return s.strip()


def exact_match(pred, truth):
    return normalize(pred) == normalize(truth)


def _tok(s):
    return normalize(s).split()


def word_f1(pred, truth):
    a = set(_tok(pred))
    b = set(_tok(truth))
    if not a or not b:
        return 1.0 if a == b else 0.0
    inter = len(a & b)
    p = inter / len(a)
    r = inter / len(b)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def answer_recall(pred, truth):
    """Fraction of ground-truth answer tokens contained in the prediction.
    1.0 means the full answer is present (correct for our purposes)."""
    a = set(_tok(pred))
    b = set(_tok(truth))
    if not b:
        return 1.0
    return len(a & b) / len(b)


def score(pred, truth):
    return {"exact": exact_match(pred, truth), "f1": word_f1(pred, truth),
            "ar": answer_recall(pred, truth)}

