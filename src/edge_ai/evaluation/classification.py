"""One full-class top-1 metric implementation, outside inference timing."""


def numpy_logits(output):
    import numpy as np

    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    logits = np.asarray(output)
    if logits.ndim != 2 or logits.shape[0] != 1 or not np.isfinite(logits).all():
        raise ValueError("reference logits must be finite with shape [1, classes]")
    return logits


def top1_correct(logits, label: int) -> int:
    if type(label) is not int or not 0 <= label < logits.shape[1]:
        raise ValueError("label outside full classifier output")
    return int(int(logits.argmax(axis=1)[0]) == label)
