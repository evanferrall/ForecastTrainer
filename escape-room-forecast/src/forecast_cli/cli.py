import typer
from pathlib import Path
import os # Import os for chdir

# Adjust the import path based on the new src structure
# Assuming train.py is in forecast_cli.training.train
from forecast_cli.training.train import train as train_model

app = typer.Typer(help="Escape Room Forecasting CLI")

@app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the configuration YAML file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    )
):
    """
    Train a new forecasting model using the specified configuration file.
    """
    typer.echo(f"Starting training with config file: {config}")
    
    # The train_model function might expect to be run from the project root
    # where 'runs/' and 'conf/' directories are typically located relative to.
    # We are in escape-room-forecast/src/forecast_cli when this runs.
    # The project root for the train_model script is escape-room-forecast/
    
    # Get the directory of the current file (cli.py)
    # src/forecast_cli/cli.py
    # current_script_dir = Path(__file__).parent 
    # src_dir = current_script_dir.parent # src/
    # project_root = src_dir.parent # escape-room-forecast/
    
    # A more robust way if the script is run via `poetry run forecast`
    # from the `escape-room-forecast` directory (which is typical for poetry projects)
    # is to assume the CWD is already the project root.
    # However, the `train_model` function from `train.py` uses relative paths like "runs/"
    # and the config path might be relative to the `escape-room-forecast` dir.
    # The `config` path is resolved to absolute by Typer's `resolve_path=True`.

    # The `train_model` function in `escape-room-forecast/src/forecast_cli/training/train.py`
    # uses paths like `os.path.join("runs", experiment_name, ...)`
    # This assumes the CWD is `escape-room-forecast`.
    # When running `poetry run forecast ...` from `escape-room-forecast`, CWD is correct.
    
    train_model(config_path=str(config)) # Pass the absolute path
    typer.echo("Training finished.")

@app.command()
def dummy():
    """
    A dummy command to test the CLI.
    """
    typer.echo("Dummy command executed!")

if __name__ == "__main__":
    app() 