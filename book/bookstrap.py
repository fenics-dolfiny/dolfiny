import argparse
import os
import pathlib
import re
import subprocess
import time

demo_files = [  # relative from demo/
    "obstacle/membrane.py",
    "obstacle/montreal.py",
    "plasticity/J2_monolithic.py",
    "plasticity/rankine.py",
    "spectral/solid_elasticity.py",
    "structural_optimisation/truss_sizing.py",
    "structural_optimisation/topopt_2d_cantilever.py",
    "structural_optimisation/topopt_3d_ge_bracket.py",
    "units/navier_stokes.py",
    "continuation/tdc_shell_naghdi_cylindrical_roof.py",
    "vsolid/solid_disp_tda.py",
]

parser = argparse.ArgumentParser("bookstrap")
parser.add_argument(
    "-f",
    "--filter",
    default=".*",
    help="Regex for filtering demos. E.g. pass '.*membrane.*' to only execute membrane demo.",
)
parser.add_argument(
    "-j",
    "--jobs",
    default=1,
    type=int,
    help="Number of parallel jobs to run when executing notebooks.",
)

args = parser.parse_args()
filter = args.filter
njobs = args.jobs

if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    raise RuntimeError("setup.py expects to be executed from the book/ directory.")


# Step 1: Copy entire demo directories to book structure
# Collect unique directories to avoid copying the same directory multiple times
demo_dirs = set(pathlib.Path(demo).parent for demo in demo_files if re.match(filter, demo))

for book_dir in sorted(demo_dirs):
    demo_dir = pathlib.Path(f"../demo/{book_dir}")

    print(f"📁 Copying directory: {demo_dir} → {book_dir}", flush=True)
    subprocess.run(["cp", "-rf", demo_dir, book_dir.parent], check=True)

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


def poll_jobs(running_jobs: list[subprocess.Popen], num: int) -> None:
    """Wait until the number of running jobs is less than or equal to num."""
    polling_interval = 0.5

    while len(running_jobs) > num:
        time.sleep(polling_interval)

        # check any crashed
        if any(p for p in running_jobs if p.poll() is not None and p.returncode < 0):
            for _p in running_jobs:
                _p.kill()

            raise RuntimeError(f"Job crashed (PID: {p.pid})")

        # Check for completed jobs and remove them from the list
        running_jobs = [p for p in running_jobs if p.poll() is None]


running_jobs = []

# Step 3: Execute notebooks
for demo in demo_files:
    notebook = pathlib.Path(demo).with_suffix(".ipynb")

    if not re.match(filter, demo):
        print(f"🚧 Executing notebook (Skipped): {notebook}", flush=True)
        continue

    if not notebook.exists():
        print(f"🚧 Executing notebook (Not found): {notebook}", flush=True)
        continue

    poll_jobs(running_jobs, njobs - 1)

    p = subprocess.Popen(
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
        ]
    )
    running_jobs.append(p)
    print(f"🚧 Executing notebook: {notebook} (PID: {p.pid})", flush=True)

poll_jobs(running_jobs, 0)  # Wait for all remaining jobs to finish

print("📖 ready!")
print(" > jupyter book start")
