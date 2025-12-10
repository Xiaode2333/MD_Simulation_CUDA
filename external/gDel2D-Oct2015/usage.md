gDel2D (GPU 2D constrained Delaunay triangulation)
==================================================

This document explains how to use the `gDel2D` code in
`external/gDel2D-Oct2015` as a library from your own C++ code. The focus
is on **feeding point sets (with optional constraints)** and reading back
the **triangle connectivity**, which you can then post‑process or export
for your own Python visualization utilities (for example
`python/plot_triangulation_python.py`).

The built‑in OpenGL visualizer (`Visualizer.*`) is *not* required and
can be ignored or disabled.


Overview
--------

Core types and API live under `external/gDel2D-Oct2015/src/gDel2D`:

- `Point2` – 2D point with coordinates `RealType _p[2]`
- `Segment` – constraint edge referencing two point indices
- `Tri` – triangle connectivity as vertex indices `int _v[3]`
- `GDel2DInput`
  - `Point2HVec pointVec;`      – input point positions
  - `SegmentHVec constraintVec;` – optional constrained edges
  - flags: `insAll`, `noSort`, `noReorder`, `profLevel`
- `GDel2DOutput`
  - `TriHVec triVec;`           – triangle list
  - `TriOppHVec triOppVec;`     – adjacency + constraint flags
  - `Point2 ptInfty;`           – artificial “infinity” point
  - `Statistics stats;`         – timings
- `class GpuDel`
  - `void compute(const GDel2DInput& input, GDel2DOutput* output);`

The GPU kernel internally appends one extra point (the “infinity” point)
at the end of the point array and builds an *output mesh on a topological
2‑sphere*. For most MD / plotting use cases you will:

1. Provide your own finite point set (and optional segment constraints).
2. Call `GpuDel::compute`.
3. Read back triangle indices from `output.triVec`, ignoring any
   triangles that reference the infinity vertex.
4. Convert the remaining triangles into the format expected by your own
   code (e.g. triangle vertex positions and metadata for CSV export).


Basic usage (no constraints)
----------------------------

Minimal example using only points (plain Delaunay triangulation):

```cpp
#include "gDel2D/GpuDelaunay.h"

void run_triangulation_example()
{
    // 1. Fill input points (host vector)
    GDel2DInput input;
    input.pointVec.clear();
    input.constraintVec.clear();

    // Example: push back your MD particle positions
    // Assume you have N points with coordinates (x[i], y[i]) in double.
    for (int i = 0; i < N; ++i) {
        Point2 p;
        p._p[0] = x[i];
        p._p[1] = y[i];
        input.pointVec.push_back(p);
    }

    // Optional tuning flags (defaults are usually fine)
    input.insAll    = false;        // incremental insert + flip
    input.noSort    = false;        // allow internal sorting
    input.noReorder = false;        // allow internal reordering
    input.profLevel = ProfDefault;  // basic timing

    // 2. Run on GPU
    GDel2DOutput output;
    GpuDel gpuDel;
    gpuDel.compute(input, &output);

    // 3. Extract triangles, skipping infinity vertex
    const int infIdx = static_cast<int>(input.pointVec.size()); // appended internally
    const TriHVec& triVec = output.triVec;

    for (int ti = 0; ti < static_cast<int>(triVec.size()); ++ti) {
        const Tri& t = triVec[ti];
        if (t._v[0] == infIdx || t._v[1] == infIdx || t._v[2] == infIdx) {
            continue; // triangle touches infinity – ignore in planar mesh
        }

        // t._v[k] are indices into the *augmented* point array.
        // If you did not enable reordering, indices 0..N-1 correspond
        // directly to your original points.
        int i0 = t._v[0];
        int i1 = t._v[1];
        int i2 = t._v[2];

        // Example: write out triangle vertex positions for CSV/plotting.
        const Point2& p0 = input.pointVec[i0];
        const Point2& p1 = input.pointVec[i1];
        const Point2& p2 = input.pointVec[i2];

        // ... convert/store (p0, p1, p2) as needed ...
    }
}
```

Notes:

- `input.pointVec` is a `thrust::host_vector<Point2>`. You fill it on
  the host, then `GpuDel::compute` automatically transfers data to the
  GPU and back.
- `GpuDel::compute` overwrites `output.triVec`, `output.triOppVec`,
  `output.ptInfty`, and timing in `output.stats`.
- The infinity vertex index is always `input.pointVec.size()` *after*
  you have pushed all your finite points (before calling `compute`).


Using constraint segments (optional)
------------------------------------

To construct a **constrained Delaunay triangulation**, provide a list of
segments referencing the point indices in `input.pointVec`:

```cpp
// After filling input.pointVec with N points:
// Add a constraint from vertex i to j
Segment s;
s._v[0] = i;
s._v[1] = j;
input.constraintVec.push_back(s);
```

Constraints are enforced by the GPU algorithm via edge‑flipping
(`GpuDelaunay.cu`); in the final mesh you can detect which triangle
edges are constrained via `output.triOppVec`:

```cpp
const TriHVec& triVec    = output.triVec;
const TriOppHVec& oppVec = output.triOppVec;

for (int ti = 0; ti < static_cast<int>(triVec.size()); ++ti) {
    const Tri& tri    = triVec[ti];
    const TriOpp& opp = oppVec[ti];

    for (int vi = 0; vi < 3; ++vi) {
        if (opp._t[vi] == -1) continue;          // boundary / no neighbour
        const bool isCons = opp.isOppConstraint(vi);
        const int  tj     = opp.getOppTri(vi);
        const int  vj     = opp.getOppVi(vi);
        // (tri,ti,vi) and (tri,tj,vj) share a constrained edge when isCons==true
    }
}
```

If you only care about triangle connectivity (for plotting or further
simulation), you may ignore `triOppVec` entirely and just export
`triVec` as described earlier.


Mapping output to existing Python visualization
-----------------------------------------------

Your existing Python visualization uses `python/particle_csv.py` and
`python/plot_triangulation_python.py`. The expected **triangulation CSV**
layout (see `load_triangulation_csv`) is:

1. First row: metadata key/value pairs, e.g.
   - `n_particles`, `n_triangles`, `box_w`, `box_h`, `draw_box`, …
2. Next `n_particles` lines: one per point
   - `x,<x>,y,<y>` or `x,<x>,y,<y>,type,<int_type>`
3. Next `n_triangles` lines: one per triangle
   - `x0,<x>,y0,<y>,x1,<x>,y1,<y>,x2,<x>,y2,<y>`

In terms of `gDel2D` output:

- Use your MD positions directly (or `input.pointVec`) for the particle
  section.
- For each triangle `t` in `output.triVec` that does *not* touch the
  infinity vertex, write one row with the three vertex coordinates:

```cpp
// Pseudocode: export a triangulation CSV compatible with plot_triangulation_csv
// Assumes you already wrote metadata and particle rows.

const int infIdx = static_cast<int>(input.pointVec.size());
const TriHVec& triVec = output.triVec;

for (int ti = 0; ti < static_cast<int>(triVec.size()); ++ti) {
    const Tri& t = triVec[ti];
    if (t._v[0] == infIdx || t._v[1] == infIdx || t._v[2] == infIdx) {
        continue; // skip infinity triangles
    }

    const Point2& p0 = input.pointVec[t._v[0]];
    const Point2& p1 = input.pointVec[t._v[1]];
    const Point2& p2 = input.pointVec[t._v[2]];

    // write: x0,p0._p[0],y0,p0._p[1],x1,p1._p[0],y1,p1._p[1],x2,p2._p[0],y2,p2._p[1]
}
```

Once you have such a CSV, you can call:

```bash
python -m python.plot_triangulation_python \
    --csv_name triangulation_snapshot.csv \
    --output_name triangulation_snapshot.png
```

This will use your existing `plot_triangulation_csv` helper to
visualize the mesh (scatter + triangle edges).


CUDA / device setup
-------------------

The sample `Main.cpp` shows typical device handling:

- Choose best device: `cutGetMaxGflopsDeviceId()`
- Set device and reset: `cudaSetDevice(deviceIdx); cudaDeviceReset();`
- After each run, call `cudaDeviceReset()` again when needed.

When embedding into your own code, you are free to:

- Reuse an existing CUDA context and skip device selection if you
  already manage it elsewhere.
- Run multiple `GpuDel::compute` calls on the same device; the class
  internally manages its device vectors and temporary memory.


Build / integration notes
-------------------------

- This project is currently built as an executable from
  `external/gDel2D-Oct2015/CMakeLists.txt`. To use it as a library:
  - Add `external/gDel2D-Oct2015/src` to your include path.
  - Add the `.cu` files from `src/gDel2D/GPU` and the `.cpp` files from
    `src/gDel2D/CPU` and the root `src` directory to your own CMake
    target, or build a static library from them.
  - You do **not** need to compile `Visualizer.cpp` if you do not use
    the OpenGL visualizer.
- `RealType` is `double` by default; define `REAL_TYPE_FP32` before
  including `CommonTypes.h` (or via compiler definitions) to switch to
  single precision.


Summary of inputs and outputs
-----------------------------

Inputs (from your code into gDel2D):

- `GDel2DInput::pointVec` – 2D point set (MD particle positions)
- `GDel2DInput::constraintVec` – optional constrained segments by
  vertex index
- Optional flags: `insAll`, `noSort`, `noReorder`, `profLevel`

Outputs (from gDel2D back to your code):

- `GDel2DOutput::triVec` – triangle list (indices into the augmented
  point array; filter out triangles touching the infinity index)
- `GDel2DOutput::triOppVec` – adjacency + constraint flags (optional)
- `GDel2DOutput::ptInfty` – coordinates of the infinity point
- `GDel2DOutput::stats` – timings, useful for profiling but not required
  for visualization

For basic usage where you “just need the triangulation triangles”, it is
enough to:

1. Fill `input.pointVec` (and optionally `input.constraintVec`),
2. Call `GpuDel::compute(input, &output)`,
3. Take `output.triVec`, drop triangles referencing the infinity vertex,
4. Convert each remaining triangle to the `(x0,y0,x1,y1,x2,y2)` format
   that your Python plotting tools expect.

