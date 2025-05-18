import importlib.metadata, inspect, textwrap, pathlib, sys

# Get neuralforecast version and nhits.py path
nf_version = importlib.metadata.version("neuralforecast")
nhits_module_path = pathlib.Path(importlib.import_module("neuralforecast.models.nhits").__file__)
print(f"neuralforecast version: {nf_version}")
print(f"nhits.py is at: {nhits_module_path}")

# Get the NHITS class and its forward method source
nhits_module = importlib.import_module("neuralforecast.models.nhits")
nhits_class = getattr(nhits_module, "NHITS")
forward_method_source = inspect.getsource(nhits_class.forward)

print("\nRelevant section in NHITS.forward() source:")
lines = textwrap.dedent(forward_method_source).splitlines()

# Try to find the specific line mentioned in the traceback, or surrounding lines
# The traceback mentioned line 412 in the file, not the method. 
# We need to read the file content directly for specific line numbers.

print("\n--- Relevant lines from nhits.py file (around line 412 if possible) ---")
problematic_line_found_in_file = False
line_context = []
TARGET_LINE_PATTERN = "forecast.repeat(1, self.h, 1)"

with open(nhits_module_path, 'r') as f:
    file_lines = f.readlines()

for i, line_content in enumerate(file_lines):
    if TARGET_LINE_PATTERN in line_content and "block_forecasts = [" in line_content:
        problematic_line_found_in_file = True
        print(f"Found pattern '{TARGET_LINE_PATTERN}' at line {i+1} in file:")
        start = max(0, i - 5)
        end = min(len(file_lines), i + 6)
        for k in range(start, end):
            print(f"{k+1:5d} | {file_lines[k].rstrip()}")
        break # Found the most likely problematic line

if not problematic_line_found_in_file:
    print(f"Could not find the exact line pattern '{TARGET_LINE_PATTERN}' leading to 'block_forecasts = [...'." )
    print("Showing source of the forward method instead:")
    for i, line in enumerate(lines):
        if TARGET_LINE_PATTERN in line and "block_forecasts = [" in line:
            print("Found pattern in method source:")
            for j in range(max(0, i-5), min(len(lines), i+5)):
                print(f"{j+1:4} (method) | {lines[j]}")
            break

print("\n--- Search for 'forecast = insample_y[:, -1:, None]' vs 'forecast = insample_y[:, -1:]' ---")
if "forecast = insample_y[:, -1:, None]" in forward_method_source:
    print("Found: forecast = insample_y[:, -1:, None]  (problematic 4D version)")
elif "forecast = insample_y[:, -1:]" in forward_method_source and "forecast.repeat(1, self.h, 1)" in forward_method_source:
    # Check if it's the specific two-liner you expected for 1.7.5
    if "forecast = insample_y[:, -1:]\n        forecast = forecast.repeat(1, self.h, 1)" in forward_method_source.replace(" ", ""):
        print("Found: forecast = insample_y[:, -1:] followed by forecast = forecast.repeat(1, self.h, 1) (expected 3D version for 1.7.5)")
    else:
        print("Found: forecast = insample_y[:, -1:] (3D version) and a forecast.repeat call, but not the exact two-line sequence.")
else:
    print("Did not find the expected forecast initialization patterns.") 