import argparse
import os
import pathlib
import subprocess

system_dependencies = ["nodejs", "npm"]
demo_files = ["obstacle/membrane.py"]  # relative from demo/


parser = argparse.ArgumentParser()
parser.add_argument("--install-dependencies", action="store_true", default=False)
args = parser.parse_args()

if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    raise RuntimeError("setup.py expects to be executed from the book/ directory.")

if args.install_dependencies:
    print("Installing dependencies...")
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-y", "install", *system_dependencies], check=True)


for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")
    print(f"♻️ {demo} → {notebook}")
    subprocess.run(["mkdir", "-p", notebook.parent], check=True)
    subprocess.run(
        ["jupytext", f"../demo/{demo}", "--to", "ipynb", "--output", notebook], check=True
    )

for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")
    print(f"🚧 {notebook}")
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
