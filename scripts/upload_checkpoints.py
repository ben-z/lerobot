import os
import time
from pathlib import Path
from typing import Optional, List
import typer
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HfHubHTTPError

app = typer.Typer()
api = HfApi()

CHECKPOINT_SUBDIR = "checkpoints"
PRETRAINED_MODEL_SUBDIR = "pretrained_model"


def get_model_dirs(base_dir: Path) -> List[Path]:
    return sorted([
        d for d in base_dir.iterdir()
        if d.is_dir()
    ])


def get_checkpoints(model_dir: Path) -> List[Path]:
    ckpt_base = model_dir / CHECKPOINT_SUBDIR
    if not ckpt_base.exists():
        return []
    return sorted([
        d for d in ckpt_base.iterdir()
        if d.is_dir() and not d.is_symlink()
    ], reverse=True)


@app.command()
def discover(echo: bool = True, detailed: bool = False, base_dir: Path = Path("outputs/train")):
    """
    Discover all models and their checkpoints.

    Parameters:
        echo: If True, print the commands to upload the checkpoints.
        detailed: If True, print each checkpoint. Otherwise, only print the model name and the number of checkpoints.
        base_dir: Base directory to search for models.
    """
    model_dirs = get_model_dirs(base_dir)

    typer.echo(f"Found {len(model_dirs)} model(s):")
    for model_dir in model_dirs:
        model_name = model_dir.name
        checkpoints = get_checkpoints(model_dir)
        if echo:
            typer.echo(f"\n#{model_name}: {len(checkpoints)} checkpoint(s)")
            typer.echo(f"python {__file__} upload '{model_dir}'")
        if detailed:
            for ckpt in checkpoints:
                if echo:
                    typer.echo(f"python {__file__} upload '{model_dir}' --checkpoint '{ckpt.name}'")
                else:
                    yield model_name, model_dir, ckpt.name
        else:
            yield model_name, model_dir, None

@app.command()
def upload(
    model_path: Path = typer.Argument(..., help="Path to model directory (e.g. outputs/train/user/model_name)"),
    checkpoint: Optional[str] = typer.Option(None, help="Specific checkpoint name to upload"),
    max_retries: int = typer.Option(10, help="Maximum number of retries"),
    backoff_factor: int = typer.Option(2, help="Backoff factor"),
    initial_backoff: int = typer.Option(10, help="Initial backoff time in seconds"),
    username: Optional[str] = typer.Option(None, help="Hugging Face user to upload to"),
):
    """
    Upload all or one checkpoint for a given model path.
    """
    if username is None:
        username = whoami()["name"]
    model_name = f"{username}/{model_path.name}"
    ckpt_dirs = get_checkpoints(model_path)

    if checkpoint:
        ckpt_dirs = [d for d in ckpt_dirs if d.name == checkpoint]
        if not ckpt_dirs:
            typer.echo(f"No checkpoint named '{checkpoint}' found in {model_path}")
            raise typer.Exit(1)

    for ckpt in ckpt_dirs:
        repo_id = f"{model_name}_{ckpt.name}"
        model_subdir = ckpt / PRETRAINED_MODEL_SUBDIR
        if not model_subdir.exists():
            typer.echo(f"Skipping {repo_id}: {PRETRAINED_MODEL_SUBDIR} not found.")
            continue

        for attempt in range(max_retries):
            try:
                typer.echo(f"[{repo_id}] Creating repo (attempt {attempt + 1})...")
                api.create_repo(repo_id=repo_id, exist_ok=True)
                typer.echo(f"[{repo_id}] Uploading from: {model_subdir}")
                api.upload_folder(folder_path=str(model_subdir), repo_id=repo_id)
                typer.echo(f"[{repo_id}] Upload complete.")
                break
            except HfHubHTTPError as e:
                if "429" in str(e):
                    wait = initial_backoff * backoff_factor ** attempt
                    typer.echo(f"[{repo_id}] Rate limited. Waiting {wait}s before retry... {e}")
                    time.sleep(wait)
                else:
                    typer.echo(f"[{repo_id}] Failed: {e}")
                    raise typer.Exit(1)
        else:
            typer.echo(f"[{repo_id}] Failed after {max_retries} attempts.")
            raise typer.Exit(1)

@app.command()
def upload_all(
    base_dir: Path = Path("outputs/train"),
    username: Optional[str] = typer.Option(None, help="Hugging Face user to upload to"),
):
    """
    Upload all checkpoints for a given user.
    """
    if username is None:
        username = whoami()["name"]

    discovered = list(discover(echo=False, base_dir=base_dir, detailed=True))
    typer.echo(f"Uploading {len(discovered)} checkpoints:")
    for model_name, model_dir, checkpoint in discovered:
        typer.echo(f"{model_name} ({model_dir}) CKPT: {checkpoint}")
    
    typer.confirm("Continue?", abort=True)
    for model_name, model_dir, checkpoint in discovered:
        typer.echo(f"Uploading {model_name} ({model_dir}) CKPT: {checkpoint}")
        upload(model_path=model_dir, checkpoint=checkpoint, username=username)


if __name__ == "__main__":
    app()