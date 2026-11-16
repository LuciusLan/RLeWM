"""Task namespaces over the existing lazy bank; legacy PushT assets stay intact."""

from pathlib import Path

from ..frame_bank import FrameLatentBank, WindowItems, build_frame_bank
from ..runtime import file_hash


def namespace_identity():
    return {"namespace_implementation_sha256": file_hash(Path(__file__))}


class SourceWindowItems(WindowItems):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if isinstance(item, list):
            return item
        namespace = self.manifest["data_config"]["dataset_id"]
        return {**item, "transition_id": f"{namespace}:{item['episode_id']}:{item['start_in_episode']}"}


class SourceFrameLatentBank(FrameLatentBank):
    def __init__(self, store, manifest, split, starts, identity):
        super().__init__(store, manifest, split, starts, identity)
        self.items = SourceWindowItems(manifest, starts)

    def to(self, device):
        return SourceFrameLatentBank(self.store.to(device), self.manifest, self.split, self.items.starts, self.identity)

    def for_evaluation(self):
        bank = super().for_evaluation()
        return SourceFrameLatentBank(bank.store, bank.manifest, bank.split, bank.items.starts, bank.identity)


def build_source_bank(adapter, manifest, normalization, split, directory, batch_size, checkpoint_rows):
    bank = build_frame_bank(adapter, manifest, normalization, split, directory, batch_size, checkpoint_rows)
    return SourceFrameLatentBank(bank.store, manifest, split, bank.items.starts, bank.identity | namespace_identity())
