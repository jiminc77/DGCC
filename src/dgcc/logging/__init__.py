"""Transition logging, immutable run registry, and asset-firewall interfaces."""
from .attempt_registry import AttemptRegistry, RegistryCorruptionError
from .asset_firewall import (
    AssetAccessError,
    AssetFirewall,
    generate_r3_r4_existence_receipt,
    load_launch_asset_manifest,
    persist_launch_receipts,
    read_launch_asset_snapshot,
)
from .code_manifest import (
    canonical_json,
    required_runtime_files,
    validate_code_manifest_bytes,
)
