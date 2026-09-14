import hashlib
import json


def generate_model_version(model):
    """
    Generate a deterministic version identifier
    from the model configuration.
    """

    model_config = {
        "class": model.__class__.__name__,
        "parameters": model.get_params(),
    }

    config_string = json.dumps(
        model_config,
        sort_keys=True,
        default=str
    )

    version_hash = hashlib.sha256(
        config_string.encode()
    ).hexdigest()

    return f"v-{version_hash[:8]}"