import os
import argparse
import sys
import datetime

# Add forecast_cli to Python path to allow direct execution and import
# This assumes the script is run from the 'escape-room-forecast' directory root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # Assuming this script is in escape-room-forecast/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from forecast_cli.utils.convert import to_onnx, onnx_to_coreml

def find_latest_run_output_dir(base_runs_dir="runs", experiment_name="escape_room_gpu"):
    """Finds the latest run directory created by train.py."""
    experiment_dir = os.path.join(base_runs_dir, experiment_name)
    if not os.path.isdir(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return None

    all_run_dirs = sorted(
        [d for d in os.listdir(experiment_dir) 
         if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith("run_")],
        key=lambda d: os.path.getmtime(os.path.join(experiment_dir, d)),
        reverse=True
    )

    if not all_run_dirs:
        print(f"Error: No run directories found in {experiment_dir}")
        return None
    
    return os.path.join(experiment_dir, all_run_dirs[0])

def main():
    parser = argparse.ArgumentParser(description="Convert a trained PyTorch model to ONNX and CoreML.")
    parser.add_argument(
        "--trained_model_path", 
        type=str, 
        help="Path to the trained PyTorch model (.pt file). If not provided, attempts to find the latest."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the ONNX and CoreML models. If not provided, uses the directory of the input model."
    )
    parser.add_argument(
        "--onnx_filename", 
        type=str, 
        default="chronos_escape_room.onnx", 
        help="Filename for the output ONNX model."
    )
    parser.add_argument(
        "--mlmodel_filename", 
        type=str, 
        default="escape_room.mlmodel", 
        help="Filename for the output CoreML model."
    )
    parser.add_argument(
        "--min_deployment_target",
        type=str,
        default="iOS17",
        help="Minimum deployment target for CoreML (e.g., iOS17, macOS14)."
    )

    args = parser.parse_args()

    trained_pt_path = args.trained_model_path
    output_conversion_dir = args.output_dir

    if not trained_pt_path:
        print("Trained model path not provided, attempting to find the latest run...")
        latest_run_dir = find_latest_run_output_dir()
        if not latest_run_dir:
            print("Could not automatically find the latest trained model. Please specify --trained_model_path.")
            return 1
        
        # Default name as per train.py and guide
        potential_model_name = "chronos_escape_room_best.pt" 
        trained_pt_path = os.path.join(latest_run_dir, potential_model_name)
        if not output_conversion_dir:
            output_conversion_dir = latest_run_dir # Save in the same directory as the .pt file
        print(f"Using latest trained model found at: {trained_pt_path}")
    elif not output_conversion_dir:
        output_conversion_dir = os.path.dirname(trained_pt_path)

    if not os.path.exists(trained_pt_path):
        print(f"Error: Trained model not found at {trained_pt_path}")
        # For simulation, if train.py hasn't run, let's create a dummy .pt file
        print(f"Creating a dummy {trained_pt_path} for simulation purposes.")
        os.makedirs(os.path.dirname(trained_pt_path), exist_ok=True)
        with open(trained_pt_path, 'w') as f:
            f.write("dummy PyTorch model data for conversion script")
    
    os.makedirs(output_conversion_dir, exist_ok=True)

    onnx_output_path = os.path.join(output_conversion_dir, args.onnx_filename)
    mlmodel_output_path = os.path.join(output_conversion_dir, args.mlmodel_filename)

    print(f"\\nStarting conversion process for: {trained_pt_path}")
    print(f"Output ONNX model will be saved to: {onnx_output_path}")
    print(f"Output CoreML model will be saved to: {mlmodel_output_path}")

    # Step 1: Convert .pt to .onnx
    # The to_onnx helper function from convert.py takes the input .pt path and the desired .onnx output path
    # It simulates the conversion and saves a dummy .onnx file.
    actual_onnx_path = to_onnx(trained_pt_path, onnx_output_path)

    if not actual_onnx_path or not os.path.exists(actual_onnx_path):
        print("ONNX conversion failed or output file not found. Aborting CoreML conversion.")
        return 1

    # Step 2: Convert .onnx to .mlmodel
    # The onnx_to_coreml helper function from convert.py takes the .onnx model path and the desired .mlmodel output path
    # It simulates the conversion and saves a dummy .mlmodel file.
    actual_mlmodel_path = onnx_to_coreml(actual_onnx_path, mlmodel_output_path, args.min_deployment_target)

    if not actual_mlmodel_path or not os.path.exists(actual_mlmodel_path):
        print("CoreML conversion failed or output file not found.")
        return 1

    print("\\nConversion process successfully simulated.")
    print(f"  Input PyTorch model: {trained_pt_path}")
    print(f"  Output ONNX model: {actual_onnx_path}")
    print(f"  Output CoreML model: {actual_mlmodel_path}")
    return 0

if __name__ == "__main__":
    # Ensure datetime is imported if not already at the top of this script
    # For this script, datetime is not directly used in main() unless for new timestamps.
    # The helper functions in convert.py might use it.
    sys.exit(main()) 