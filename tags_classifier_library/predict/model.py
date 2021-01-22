import logging
import os
from dataclasses import dataclass
from typing import List

import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    name: str
    group: str
    path: str
    group_path: str


_model_cache = {}


def get_model(path, force_load=False):
    """Get Tensorflow model."""
    global _model_cache
    if path in _model_cache and not force_load:
        return _model_cache[path]
    model = tf.keras.models.load_model(path)
    _model_cache[path] = model
    return model


def inspect_model(models_path, model_groups) -> List[ModelInfo]:
    """
    Inspect models in a given path
    """
    if not os.path.isdir(models_path):
        return None

    logger.info(f"model dir: {models_path}")
    logger.info(f"list contents: {os.listdir(models_path)}")

    models = []

    for group in model_groups:
        group_path = os.path.join(models_path, group)

        for model in os.listdir(group_path):
            path = os.path.join(group_path, model.replace("/", "\\/"))
            if os.path.isdir(path):
                # some models are nested, so we need to find out and adjust
                # the model name and path
                dir_contents = os.listdir(path)
                if len(dir_contents) == 1:
                    new_path = os.path.join(path, dir_contents[0])
                    if not os.path.isdir(new_path):
                        logger.warning(f"Model skipped: {model}, wrong directory structure.")
                        break

                    model = "/".join([model, dir_contents[0]])
                    path = new_path

                info = ModelInfo(
                    name=model,
                    group=group,
                    path=path,
                    group_path=group_path,
                )
                models.append(info)

    return models
