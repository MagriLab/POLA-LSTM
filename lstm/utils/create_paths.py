from pathlib import Path


def make_folder_filepath(model_path: Path, foldername: str) -> Path:
    """makes a folder in the model filepath

    Args:
        model_path (Path): folder of model

    Returns:
        Path: image filepath
    """
    folder_filepath = model_path / foldername 
    folder_filepath.mkdir(parents=True, exist_ok=True)
    return folder_filepath
