import argparse
import os
import pathlib
import re
import subprocess

demo_files = [  # relative from demo/
    "obstacle/membrane.py",
    "obstacle/montreal.py",
    "structural_optimisation/truss_sizing.py",
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


for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")

    if not re.match(filter, demo):
        print(f"♻️ {demo} → {notebook} (Skipped)", flush=True)
        continue

    print(f"♻️ {demo} → {notebook}", flush=True)
    subprocess.run(["mkdir", "-p", notebook.parent], check=True)
    subprocess.run(
        ["jupytext", f"../demo/{demo}", "--to", "ipynb", "--output", notebook], check=True
    )

for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")

    if not re.match(filter, demo):
        print(f"🚧 {notebook} (Skipped)", flush=True)
        continue

    print(f"🚧 {notebook}", flush=True)
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--inplace",
            "--clear-output",
            notebook,
        ],
        check=True,
    )

print("📖 ready!")
print(" > jupyter book start")
