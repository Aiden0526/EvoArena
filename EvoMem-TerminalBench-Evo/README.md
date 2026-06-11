# EvoMem-TerminalBench-Evo

This folder contains the EvoMem implementation for Terminal-Bench-Evo experiments. It runs **Terminus2** on sequential evolution benchmarks with optional **EvoMem** (chain-scoped memory across prototype and variant tasks).

| Component | Path | Role |
|-----------|------|------|
| Dataset | `Terminal-Bench-Evo/` | Task chains (prototype + `*-EVO-*` variants) |
| Runner | `harbor_EvoMem/` | Agents, scripts, and EvoMem integration |
| Harbor | `harbor/` | Submodule — benchmark runtime |

Experiments:

- `terminus2_evomem` — Terminus2 + EvoMem
- `terminus2_baseline` — Terminus2 without EvoMem

---

## 1. Dataset

Benchmark tasks are **not** in this git repo. Download **Terminal-Bench-Evo** from Hugging Face and place it at this folder root:

**https://huggingface.co/datasets/wufeiwu/Terminal-Bench-Evo**

Expected layout:

```text
EvoMem-TerminalBench-Evo/
  Terminal-Bench-Evo/    ← dataset checkout
  harbor_EvoMem/
  harbor/
```

Runners default to `<repo-root>/Terminal-Bench-Evo`. Override with `HARBOR_EVOMEM_DATASET` or `--dataset`.

---

## 2. Setup

From the EvoArena repository root:

```bash
cd EvoMem-TerminalBench-Evo
git submodule update --init --recursive
```

Install the Harbor submodule (see `harbor/README.md` for upstream details), then install the EvoMem runner:

```bash
cd harbor_EvoMem
python -m pip install -e .
```

**Requirements:** Python 3.11+, Docker, tmux, and a `harbor` CLI on your `PATH`.

---

## 3. Configure the LLM

```bash
cd harbor_EvoMem
cp scripts/terminus2_llm.env.example scripts/terminus2_llm.env
```

Edit `scripts/terminus2_llm.env`:

```bash
export LLM_API_KEY="..."
export LLM_MODEL="..."
export LLM_BASE_URL="..."
```

This file is git-ignored. All Terminus2 and EvoMem summarizer calls read from it.

---

## 4. Run experiments

From `harbor_EvoMem/` (dataset defaults to `../Terminal-Bench-Evo`):

**EvoMem:**

```bash
scripts/launch_terminus2_evomem.sh
```

**Baseline:**

```bash
scripts/launch_terminus2_baseline.sh
```

Common flags:

```bash
--parallel 4
--max-chains 30
--chains "bn-fit-modify adaptive-rejection-sampler"
--trials-dir /path/to/runs
--dataset /path/to/Terminal-Bench-Evo
```

Launches run in tmux in the background.

---

## 5. Monitor and stop

```bash
cd harbor_EvoMem
scripts/status.py --variant terminus2_evomem --watch
scripts/status.py --variant terminus2_baseline --detail

scripts/kill_runs.sh --variant terminus2_evomem
scripts/kill_runs.sh --variant terminus2_baseline
```

---

## Repository layout

```text
EvoMem-TerminalBench-Evo/
├── README.md                 ← you are here
├── Terminal-Bench-Evo/       ← HF dataset (git-ignored)
├── harbor_EvoMem/
│   ├── harbor_EvoMem/        Python package
│   └── scripts/              launchers, run_chain, status
└── harbor/                   git submodule
```
