#!/usr/bin/env python3
"""Verify visibility shader staging and stale asset detection."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("visibility_compiler", ROOT / "tools/compile_vulkan_visibility_shaders.py")
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


class VisibilityShaderBuildTest(unittest.TestCase):
    def test_failed_build_keeps_previous_assets_and_check_detects_stale_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for _, name, _ in compiler.VARIANTS:
                (output / name).write_bytes(b"previous " + name.encode())
            original = {name: (output / name).read_bytes() for _, name, _ in compiler.VARIANTS}
            calls = 0

            def fake_compile(command, check):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise subprocess.CalledProcessError(1, command)
                target = Path(command[command.index("-o") + 1])
                digest = hashlib.sha256("|".join(command[:-2]).encode()).digest()
                target.write_bytes(b"\x03\x02\x23\x07" + digest)

            args = ["visibility-build", "--output-dir", directory]
            with mock.patch.object(sys, "argv", args), mock.patch.object(compiler.subprocess, "run", fake_compile):
                with self.assertRaises(subprocess.CalledProcessError):
                    compiler.main()
            self.assertEqual(original, {name: (output / name).read_bytes() for _, name, _ in compiler.VARIANTS})

            def successful_compile(command, check):
                target = Path(command[command.index("-o") + 1])
                digest = hashlib.sha256("|".join(command[:-2]).encode()).digest()
                target.write_bytes(b"\x03\x02\x23\x07" + digest)

            with mock.patch.object(sys, "argv", args), mock.patch.object(compiler.subprocess, "run", successful_compile):
                compiler.main()
            check_args = args + ["--check"]
            with mock.patch.object(sys, "argv", check_args), mock.patch.object(compiler.subprocess, "run", successful_compile):
                compiler.main()
                (output / compiler.VARIANTS[-1][1]).write_bytes(b"stale")
                with self.assertRaises(SystemExit):
                    compiler.main()


if __name__ == "__main__":
    unittest.main()
