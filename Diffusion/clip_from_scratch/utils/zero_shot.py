from typing import Callable, Sequence, Union

import torch
import torch.nn as nn


def build_zero_shot_classifier(
        model: nn.Module,
        tokenizer: Callable,
        classnames: Sequence[str],
        templates: Sequence[Union[str, Callable]],
        device: str,
):
    """
    Build a zero-shot classifier based on the given model and classnames and templates.
    :param model: CLIP model
    :param tokenizer: tokenizer for text encoding
    :param classnames: list of classnames
    :param templates: list of templates, support python format
    :param device: device to run the model on
    :return: a matrix of class embeddings, shape (num_templates, num_classes, model_dim)
    """
    assert len(classnames) > 0 and len(templates) > 0, "classnames and templates must not be empty"
    is_format = isinstance(templates[0], str)
    num_classes = len(classnames)
    num_templates = len(templates)

    with torch.no_grad():
        texts = [template.format(c) if is_format else template(c) for c in classnames for template in templates]
        texts = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(texts, normalize=True)
        class_embeddings = class_embeddings.reshape(num_classes, num_templates, -1).mean(1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)

    return class_embeddings.T


def zero_shot_accuracy(labels, logits):
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().sum() / len(labels)
    return acc