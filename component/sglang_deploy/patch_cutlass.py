#!/usr/bin/env python3
"""Apply the CUTLASS compatibility fix required by this SGLang version."""

import importlib.util
import inspect
from pathlib import Path

from cutlass._mlir.dialects import llvm


class CutlassCompatibilityPatcher:
    def __init__(self) -> None:
        module = importlib.util.find_spec("cutlass.cutlass_dsl.tvm_ffi_provider")
        if module is None or module.origin is None:
            raise RuntimeError("CUTLASS DSL provider module was not found")
        self.path = Path(module.origin)

    def apply(self) -> bool:
        source = self.path.read_text(encoding="utf-8")
        original_source = source
        supports_data = "data" in inspect.signature(
            llvm.mlir_global_dtors
        ).parameters
        call_without_data = """                global_dtors = llvm.mlir_global_dtors(
                    dtors=[],
                    priorities=[],
                )"""
        call_with_data = """                global_dtors = llvm.mlir_global_dtors(
                    dtors=[],
                    priorities=[],
                    data=[],
                )"""
        data_markers = (
            '        global_dtors.attributes["data"] += [ir.UnitAttr.get()]\n',
            '        global_dtors.attributes["data"] += '
            '[ir.Attribute.parse("#llvm.zero")]\n',
        )
        data_marker = data_markers[1]
        data_attribute = 'global_dtors.attributes["data"] += ['

        if supports_data:
            source = source.replace(call_without_data, call_with_data, 1)
            if data_markers[0] in source:
                source = source.replace(data_markers[0], data_marker, 1)
        else:
            source = source.replace(call_with_data, call_without_data, 1)
            for marker in data_markers:
                source = source.replace(marker, "", 1)

        expected_call = call_with_data if supports_data else call_without_data
        if expected_call not in source:
            raise RuntimeError(f"Unexpected CUTLASS provider layout: {self.path}")
        if supports_data and data_attribute not in source:
            raise RuntimeError(f"CUTLASS destructor data marker is missing: {self.path}")

        changed = source != original_source
        if changed:
            self.path.write_text(source, encoding="utf-8")
        return changed


if __name__ == "__main__":
    changed = CutlassCompatibilityPatcher().apply()
    result = "applied" if changed else "already present"
    print("CUTLASS compatibility patch", result)
