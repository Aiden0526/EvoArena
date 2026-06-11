"""Storage utilities for patch-centered memory history."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


class PatchStore:
    """Persist patch records and global-graph cache sidecars."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def sample_dir(self, sample_id: str) -> Path:
        sample_dir = self.root_dir / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    def patches_dir(self, sample_id: str) -> Path:
        path = self.sample_dir(sample_id) / "patches"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def global_graph_dir(self, sample_id: str) -> Path:
        path = self.sample_dir(sample_id) / "global_graph"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def global_graph_paths(self, sample_id: str) -> tuple[str, str, str]:
        graph_dir = self.global_graph_dir(sample_id)
        return (
            str(graph_dir / f"memory_cache_sample_{sample_id}.pkl"),
            str(graph_dir / f"retriever_cache_sample_{sample_id}.pkl"),
            str(graph_dir / f"retriever_cache_embeddings_sample_{sample_id}.npy"),
        )

    def build_status_path(self, sample_id: str) -> Path:
        return self.sample_dir(sample_id) / "build_status.json"

    def save_build_status(self, sample_id: str, status: Dict) -> None:
        self.build_status_path(sample_id).write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    def load_build_status(self, sample_id: str) -> Optional[Dict]:
        path = self.build_status_path(sample_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding='utf-8'))

    def patch_index_records_path(self, sample_id: str) -> Path:
        return self.sample_dir(sample_id) / "patch_index_records.jsonl"

    def patch_retriever_paths(self, sample_id: str) -> tuple[str, str]:
        sample_dir = self.sample_dir(sample_id)
        return (
            str(sample_dir / "patch_retriever.pkl"),
            str(sample_dir / "patch_retriever_embeddings.npy"),
        )

    def save_patch(self, sample_id: str, patch_record: Dict) -> Path:
        patch_path = self.patches_dir(sample_id) / f"{patch_record['patch_id']}.json"
        patch_path.write_text(json.dumps(patch_record, ensure_ascii=False, indent=2), encoding="utf-8")
        return patch_path

    def load_patch(self, sample_id: str, patch_id: str) -> Optional[Dict]:
        patch_path = self.patches_dir(sample_id) / f"{patch_id}.json"
        if not patch_path.exists():
            return None
        return json.loads(patch_path.read_text(encoding="utf-8"))

    def clear_patch_data(self, sample_id: str) -> None:
        """Delete all patch-specific files for a sample, keeping the global graph intact."""
        sample_dir = self.root_dir / f"sample_{sample_id}"
        if not sample_dir.exists():
            return
        # Remove patch index
        idx_path = self.patch_index_records_path(sample_id)
        if idx_path.exists():
            idx_path.unlink()
        # Remove patch retriever cache
        pkl, npy = self.patch_retriever_paths(sample_id)
        for p in (pkl, npy):
            if Path(p).exists():
                Path(p).unlink()
        # Remove individual patch JSON files
        patches_dir = sample_dir / "patches"
        if patches_dir.exists():
            shutil.rmtree(patches_dir)

    def clear_global_graph_cache(self, sample_id: str) -> None:
        """Delete global-graph cache artifacts for a sample."""
        sample_dir = self.root_dir / f"sample_{sample_id}"
        if not sample_dir.exists():
            return
        graph_dir = sample_dir / "global_graph"
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
        build_status_path = self.build_status_path(sample_id)
        if build_status_path.exists():
            build_status_path.unlink()

    def load_all_patches(self, sample_id: str) -> List[Dict]:
        records = []
        for patch_path in sorted(self.patches_dir(sample_id).glob("patch_*.json")):
            records.append(json.loads(patch_path.read_text(encoding="utf-8")))
        return records

    def append_patch_index_record(self, sample_id: str, index_record: Dict) -> None:
        path = self.patch_index_records_path(sample_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    def load_patch_index_records(self, sample_id: str) -> List[Dict]:
        path = self.patch_index_records_path(sample_id)
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


    def save_patch_index_records(self, sample_id: str, records: List[Dict]) -> None:
        path = self.patch_index_records_path(sample_id)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
