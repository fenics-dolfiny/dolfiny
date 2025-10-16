import argparse
import os
import pathlib
import re
import subprocess

demo_files = [  # relative from demo/
    "obstacle/membrane.py",
    "obstacle/montreal.py",
    "spectral/solid_elasticity.py",
    "structural_optimisation/truss_sizing.py",
    "structural_optimisation/topopt_2d_cantilever.py",
    "structural_optimisation/topopt_3d_ge_bracket.py",
]

parser = argparse.ArgumentParser("bookstrap")
parser.add_argument(
    "-f",
    "--filter",
    default=".*",
    help="Regex for filtering demos. E.g. pass '.*membrane.*' to only execute membrane demo.",
)
args = parser.parse_args()
filter = args.filter

if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    raise RuntimeError("setup.py expects to be executed from the book/ directory.")


# Step 1: Copy entire demo directories to book structure
# Collect unique directories to avoid copying the same directory multiple times
demo_dirs = set()
for demo in demo_files:
    if re.match(filter, demo):
        demo_dirs.add(pathlib.Path(demo).parent)

for book_dir in sorted(demo_dirs):
    demo_dir = pathlib.Path(f"../demo/{book_dir}")

    print(f"📁 Copying directory: {demo_dir} → {book_dir}", flush=True)
    subprocess.run(["mkdir", "-p", str(book_dir)], check=True)

    # Copy all files from demo directory to book directory
    if demo_dir.exists():
        for file in demo_dir.iterdir():
            if file.is_file():
                subprocess.run(["cp", str(file), str(book_dir)], check=True)

# Step 2: Convert Python scripts to notebooks
for demo in demo_files:
    script = pathlib.Path(demo)
    notebook = script.with_suffix(".ipynb")

    if not re.match(filter, demo):
        print(f"♻️  Converting script (Skipped): {script} → {notebook}", flush=True)
        continue

    if not script.exists():
        print(f"♻️  Converting script (Not found): {script} → {notebook}", flush=True)
        continue

    print(f"♻️  Converting script to notebook: {script} → {notebook}", flush=True)
    subprocess.run(
        ["jupytext", str(script), "--to", "ipynb", "--quiet", "--output", str(notebook)],
        check=True,
    )

# Step 3: Execute notebooks
for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")

    if not re.match(filter, demo):
        print(f"🚧 Executing notebook (Skipped): {notebook}", flush=True)
        continue

    if not notebook.exists():
        print(f"🚧 Executing notebook (Not found): {notebook}", flush=True)
        continue

    print(f"🚧 Executing notebook: {notebook}", flush=True)
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--inplace",
            "--clear-output",
            "--log-level=WARN",
            str(notebook),
        ],
        check=True,
    )

print("📖 ready!")
print(" > jupyter book start")
