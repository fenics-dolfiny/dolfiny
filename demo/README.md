# Guidelines for writing dolfiny demos

This document defines rules and best practices for creating high-quality, educational, and
maintainable demos for `dolfiny`. Follow these when writing a new demo or reviewing an existing one.

## Rules for a good demo

1. **Jupyter-compatible Python scripts**
   Write demos as standard Python scripts (`.py`) using cell markers (`# %%` and `# %% [markdown]`).
   This format runs as a plain script and converts cleanly to Jupyter Notebooks or MyST documentation.

2. **Opening markdown: title, purpose, and prerequisites**
   Every demo must begin with a `[markdown]` cell containing:
   - A descriptive title (`# Title in sentence case`).
   - An introductory paragraph (not a bullet list) that states what problem is being solved and what
     the demo demonstrates. Write it as connected prose, not an itemised checklist.
   - If the demo builds on a previous one, include a `note` directive linking to it
     (see `obstacle/montreal.py` for an example).
   - An **"In particular, this demo emphasizes:"** list of 3–5 bullet points placed immediately
     after the introductory paragraph (and note, if any). Each bullet names one key technique,
     dolfiny feature, or mathematical concept the reader will encounter. Write each as a noun
     phrase, not a full sentence, and keep it to one line.

3. **Mathematical and physical context**
   Do not just write code. Explain the physics and mathematics, leading the reader from the
   continuous problem through to the discrete form actually implemented. Provide:
   - The constitutive model, strong form, or energy functional.
   - The weak/variational form that the code implements.
   - Equations in display math (`$$ ... $$`) and inline math (`$...$`) as appropriate.
   - Inline DOI citations for key references, e.g. `(see https://doi.org/...)`.
     **Do not add a `## References` section** — MyST generates it automatically from inline links.

4. **Narrative continuity**
   The documentation should read as a coherent explanation, not as a sequence of isolated
   heading + equation blocks. Equations should be introduced by the sentence that motivates them
   and followed by the sentence that interprets them. Use connective language so each paragraph
   leads naturally to the next. A good test: remove all headings — the text should still make sense
   as a narrative.

5. **Logical structure and cell pacing**
   Break code into logical cells using `# %%`. A standard flow:
   - **Imports**: grouped as MPI/PETSc, basix/dolfinx/ufl, standard libraries, dolfiny.
   - **Parameters and mesh**: geometry, material properties, mesh generation.
   - **Function spaces and functions**.
   - **Variational forms and boundary conditions**: directly translate the math from the header.
   - **Solver setup and execution**.
   - **Post-processing and visualisation**: export XDMF/VTK or plot with pyvista/matplotlib.

   Add a new `# %% [markdown]` cell only when a specific section of code truly needs its own
   explanation — not after every few lines.

6. **Cell tags for notebook rendering**
   Use Jupytext-style tags to control cell visibility in the rendered book:
   - All code cells must hide their source in the rendered notebook. Use
     `# %% tags=["hide-input"]` by default.
   - Show cell output only when it directly supports the surrounding explanation, such as a
     relevant plot, table, or short diagnostic that the demo text discusses.
   - Use `# %% tags=["hide-output"]` for cells with verbose or uninformative output
     (gmsh logs, solver iterations, routine stdout/stderr).
   - Both tags can be combined: `# %% tags=["hide-input", "hide-output"]`.

7. **Figure labels and captions**
   Every code cell that produces a pyvista or matplotlib plot output must carry a MyST `label` and
   `caption` directly inside the cell, immediately after the `# %% tags=[...]` marker:
   ```python
   # %% tags=["hide-input"]
   # | label: fig-my-plot
   # | caption: |
   #   One-sentence description of what is shown.
   ```
   - The label must start with `fig-` and use kebab-case.
   - The caption is a single sentence (or two at most) describing the content of the figure.
     Do not explain axes or restate surrounding maths — just say what is being shown.
   - Use the block-scalar form (`caption: |` with continuation lines prefixed `#   `) when the
     caption exceeds 100 characters; otherwise a single `# | caption: ...` line is fine.

8. **Contextualising visible output**
   Every cell whose output is rendered in the book (i.e. does *not* carry `hide-output`) must be
   contextualised for the reader. Specifically:
   - **Figures and plots**: the `label` / `caption` block (Rule 7) is sufficient — no additional
     `[markdown]` cell is needed for plot output.
   - **Printed / diagnostic output**: accompany the cell with a `[markdown]` cell that states what
     the numbers or text mean and why they are worth showing (e.g. convergence values,
     surface-area comparison, dimensionless parameters). This may appear immediately before or
     after the code cell, or as the closing sentence of the preceding paragraph — whichever reads
     most naturally. It must not merely restate the variable name or repeat information already
     obvious from the surrounding maths.

9. **Cleanliness and reproducibility**
   - No magic numbers. Use named variables with a comment giving physical units.
   - Follow PEP-8 (`black` and `flake8`); maximum line length is 100 characters.
   - The script must exit with code `0` without unhandled warnings.

10. **Avoid describing the obvious**
   Do not explain FEniCS/dolfiny API mechanics. Comment on the *mathematical choice* (which function
   space, which quadrature rule, why) rather than the *how* (what the function call does).
   Do not restate the same fact in two places.

11. **LaTeX conventions**
   - Section headings use **sentence case**: capitalise only the first word and proper nouns.
   - Use `\text{tr}`, `\text{dev}`, `\text{sym}` (not `\mathrm{...}`) for named operators.
   - Use `\,\text{d}x` for integration measures.
   - Keep each `$...$` or `$$...$$` block on a single comment line if possible. A display math
     block broken across Python comment lines can fail to render in MyST.

## Registering a new demo

When you create a new demo, register it in several places to ensure it is tested and rendered:

1. **`.gitlab-ci.yml`**:
   - Add the script to the `SCRIPTS` block of the relevant job (e.g. `obstacle:`, `spectral:`).
     This makes the CI pipeline run the demo to catch regressions.
   - If the demo should appear in the Jupyter book, also add it to the `REGEX` list under
     the `bookstrap` job.

2. **`book/bookstrap.py`**:
   - Add the relative path (e.g. `"obstacle/my_new_demo.py"`) to the `demo_files` list.
     The bookstrap script converts the script to `.ipynb` format during CI.

3. **`book/myst.yml`**:
   - Add the generated notebook path (`.ipynb`) to the `toc` keys so it appears in the
     documentation site's navigation menu.
