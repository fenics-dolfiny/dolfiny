import os
import pathlib
import subprocess

demo_files = ["obstacle/membrane.py"]  # relative from demo/

if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    raise RuntimeError("setup.py expects to be executed from the book/ directory.")


for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")
    print(f"♻️ {demo} → {notebook}", flush=True)
    subprocess.run(["mkdir", "-p", notebook.parent], check=True)
    subprocess.run(
        ["jupytext", f"../demo/{demo}", "--to", "ipynb", "--output", notebook], check=True
    )

for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")
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
