# EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments

### Benchmarking and improving LLM agents under persistent environment evolution.

[![Paper](https://img.shields.io/badge/arXiv-2606.13681-b31b1b.svg)](https://arxiv.org/pdf/2606.13681)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://aiden0526.github.io/EvoArena/)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Aiden0526/EvoArena)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/collections/Aiden0526/evoarena)

> **Status.** This repository is being released progressively. The main experiment folders are
> `EvoMem-PersonaMem-Evo/`, `EvoMem-Terminal-Bench-Evo/`, and `EvoMem-SWE-Chain-Evo/`.

<p align="center">
  <img src="assets/evoarena_fig1.png" width="92%">
</p>


## What Is EvoArena?

Most agent benchmarks evaluate a fixed environment snapshot. Real deployments are messier: workflows change, codebases evolve, dependencies move, validation rules tighten, and user preferences shift over time. **EvoArena** evaluates whether LLM agents can remain reliable under this kind of **persistent environment evolution**.

EvoArena models each environment as a chain of progressively evolving releases. The high-level setting remains connected across releases, but the executable workflow, repository state, or user preference state changes. Agents must solve the current task while avoiding stale behavior learned from earlier versions.

<p align="center">
  <img src="assets/evoarena_overview.png" width="92%">
</p>

EvoArena covers three complementary domains:

| Benchmark | Environment Evolution | Base Agent | What It Tests |
| --- | --- | --- | --- |
| **Terminal-Bench-Evo** | Executable terminal workflows evolve through interface, path, toolchain, validation, and policy changes. | Terminus 2 | Whether agents adapt terminal strategies across workflow releases. |
| **SWE-Chain-Evo** | Repository states evolve through chronological software milestones. | OpenHands | Whether agents solve new code requirements without regressing prior behavior. |
| **PersonaMem-Evo** | User preferences evolve across long interaction histories. | A-Mem | Whether memory agents track preference updates, conflicts, and temporal trajectories. |

## Dataset Composition

<p align="center">
  <img src="assets/evoarena_dataset_composition.png" width="72%">
</p>

EvoArena reports both **step accuracy** and **chain accuracy**. Step accuracy measures whether an agent solves an individual evolved task instance. Chain accuracy is stricter: a chain is correct only when the agent succeeds across the required sequence of related evolutionary steps.

## Persistent Environment Evolution

### Terminal-Bench-Evo

Terminal-Bench-Evo starts from Terminal-Bench tasks and turns each one into a discrete workflow-version chain. The same high-level terminal objective is preserved, but later releases may change deployment mechanisms, paths, branches, dependencies, permissions, or validation logic.

<p align="center">
  <img src="assets/terminal-bench.png" width="88%">
</p>

### SWE-Chain-Evo

SWE-Chain-Evo evaluates agents on software repositories whose code states evolve through milestone updates. At each step, the agent receives the current accumulated repository state and a localized software requirement, then produces a patch evaluated by target and regression tests.

<p align="center">
  <img src="assets/swe-evo.png" width="88%">
</p>

### PersonaMem-Evo

PersonaMem-Evo evaluates long-horizon memory under evolving user preferences. Questions require agents to infer current preferences from long interaction histories, resolve conflicts, and reason about how earlier preferences were revised.

<p align="center">
  <img src="assets/persona-mem-example.png" width="88%">
</p>

## EvoMem

**EvoMem** is a lightweight, git-like memory paradigm for evolving environments. Instead of storing only the latest memory state, EvoMem records meaningful memory updates as patches. Each patch captures what changed, why it changed, and which evidence supported the update. At inference time, agents can retrieve the latest memory together with relevant historical patches when a task depends on overwritten, conflicting, or version-specific information.

<p align="center">
  <img src="assets/evomem_framework.png" width="88%">
</p>

EvoMem is designed as a wrapper around existing agents rather than a replacement for them. In this repository, EvoMem is integrated with:

- **A-Mem** for PersonaMem-Evo.
- **Terminus 2** for Terminal-Bench-Evo.
- **OpenHands** for SWE-Chain-Evo.

## Repository Structure

```text
EvoArena/
├── EvoMem-PersonaMem-Evo/       # A-Mem + EvoMem on PersonaMem-Evo
├── EvoMem-Terminal-Bench-Evo/   # Terminus 2 + EvoMem on Terminal-Bench-Evo
├── EvoMem-SWE-Chain-Evo/        # OpenHands + EvoMem on SWE-Chain-Evo
├── assets/                      # README figures
└── README.md
```

## Running Experiments

Each main experiment folder is self-contained and will include its own detailed `README.md`, environment setup, scripts, and command examples. The top-level README only provides the entry points.

### PersonaMem-Evo

```bash
cd EvoMem-PersonaMem-Evo
# See EvoMem-PersonaMem-Evo/README.md for setup, data preparation, and evaluation.
```

This folder contains the currently released EvoMem implementation for preference-memory evolution with A-Mem.

### Terminal-Bench-Evo

```bash
cd EvoMem-Terminal-Bench-Evo
# See EvoMem-Terminal-Bench-Evo/README.md for Terminus 2 setup and evaluation.
```

This folder will contain the Terminal-Bench-Evo evaluation harness and EvoMem wrapper for chain-scoped terminal patch memory.

### SWE-Chain-Evo

```bash
cd EvoMem-SWE-Chain-Evo
# See EvoMem-SWE-Chain-Evo/README.md for OpenHands setup and evaluation.
```

This folder will contain the SWE-Chain-Evo evaluation harness and EvoMem wrapper for software patch memory.

## Data

The EvoArena dataset will be released on Hugging Face:

- **Dataset:** [https://huggingface.co/datasets/Aiden0526/EvoArena](https://huggingface.co/datasets/Aiden0526/EvoArena)

If you use a specific benchmark subset, please also check the corresponding experiment folder for any additional preprocessing or environment-packaging instructions.

## Links

- **Paper:** [https://arxiv.org/pdf/2606.13681](https://arxiv.org/pdf/2606.13681)
- **Project page:** [https://aiden0526.github.io/EvoArena/](https://aiden0526.github.io/EvoArena/)
- **Code:** [https://github.com/Aiden0526/EvoArena](https://github.com/Aiden0526/EvoArena)
- **Dataset:** [https://huggingface.co/datasets/Aiden0526/EvoArena](https://huggingface.co/collections/Aiden0526/evoarena)

## Citation

If you find EvoArena or EvoMem useful, please cite our paper:

```bibtex
@article{xu2026evoarena,
  title   = {EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments},
  author  = {Jundong Xu and Qingchuan Li and Jiaying Wu and Yihuai Lan and Shuyue Stella Li and Huichi Zhou and Bowen Jiang and Lei Wang and Jun Wang and Anh Tuan Luu and Caiming Xiong and Hae Won Park and Bryan Hooi and Zhiyuan Hu},
  journal = {arXiv preprint arXiv:2606.13681},
  year    = {2026}
}
```

## License

The license will be added with the public release. Please check the final repository license before using the code or data in downstream projects.
