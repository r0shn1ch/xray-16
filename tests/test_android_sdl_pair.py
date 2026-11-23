"""Prevent mixing the launcher's SDL Java classes with another native SDL."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_sdl_java", ROOT / "android/apk/stage_sdl_java.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


def header(path, version):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"#define SDL_{name} {value}" for name, value in
        zip(("MAJOR_VERSION", "MINOR_VERSION", "PATCHLEVEL"), version)), encoding="utf-8")
    return path


def source(path, version, api):
    header(path / "include/SDL_version.h", version)
    java = path / "android-project/app/src/main/java/org/libsdl/app"
    java.mkdir(parents=True)
    (java / "SDLActivity.java").write_text("\n".join(f"private static final int SDL_{name}_VERSION = {value};"
        for name, value in zip(("MAJOR", "MINOR", "MICRO"), version)), encoding="utf-8")
    (java / (api + ".java")).write_text(api, encoding="utf-8")
    return path


class AndroidSdlPairTest(unittest.TestCase):
    def test_stale_sdk_selects_native_version_and_replaces_whole_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = header(root / "prefix/include/SDL2/SDL_version.h", (2, 30, 2))
            stale = source(root / "sdk/SDL", (2, 30, 0), "OldNativeApi")
            correct = source(root / "build/renderer-deps/SDL-2.30.2", (2, 30, 2), "NewNativeApi")
            project = root / "project"
            destination = project / "app/src/main/java/org/libsdl/app"
            destination.mkdir(parents=True)
            (destination / "OldNativeApi.java").write_text("stale", encoding="utf-8")
            engine = project / "app/src/main/java/org/openxray/app/Launcher.java"
            engine.parent.mkdir(parents=True)
            engine.write_text("engine", encoding="utf-8")
            result = pair.stage(project, [native], [stale], repo=root)
            self.assertEqual(result["version"], "2.30.2")
            self.assertEqual(pair.java_version(destination / "SDLActivity.java"), (2, 30, 2))
            self.assertTrue((destination / "NewNativeApi.java").is_file())
            self.assertFalse((destination / "OldNativeApi.java").exists())
            self.assertEqual(engine.read_text(), "engine")
            self.assertEqual(pair.java_version(stale / "android-project/app/src/main/java/org/libsdl/app/SDLActivity.java"), (2, 30, 0))
            self.assertTrue((correct / "include/SDL_version.h").is_file())

    def test_unknown_native_source_rejects_without_relabeling_stale_java(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = header(root / "prefix/SDL_version.h", (2, 30, 2))
            stale = source(root / "sdk", (2, 30, 0), "OldApi")
            project = root / "project"
            (project / "app").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "No SDL 2.30.2 Java source"):
                pair.stage(project, [native], [stale], repo=root)
            self.assertFalse((project / "sdl-java-provenance.json").exists())
            self.assertEqual(pair.java_version(stale / "android-project/app/src/main/java/org/libsdl/app/SDLActivity.java"), (2, 30, 0))

    def test_mixed_abi_versions_cannot_share_one_java_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            headers = [header(root / "arm32/SDL_version.h", (2, 30, 0)),
                       header(root / "arm64/SDL_version.h", (2, 30, 2))]
            with self.assertRaisesRegex(ValueError, "ABIs use different SDL versions"):
                pair.stage(root / "project", headers, repo=root)

    def test_native_build_cache_supplies_the_matching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = header(root / "prefix/SDL_version.h", (2, 30, 2))
            correct = source(root / "native-source", (2, 30, 2), "NativeApi")
            build = root / "native-build"
            build.mkdir()
            (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={correct.as_posix()}\n", encoding="utf-8")
            project = root / "project"
            (project / "app").mkdir(parents=True)
            pair.stage(project, [native], builds=[build], repo=root)
            self.assertEqual(pair.java_version(project / "app/src/main/java/org/libsdl/app/SDLActivity.java"), (2, 30, 2))

    def test_staging_cannot_overwrite_the_dependency_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = source(Path(directory) / "SDL", (2, 30, 2), "NativeApi")
            with self.assertRaisesRegex(ValueError, "must be separate"):
                pair.stage(root / "android-project", [root / "include/SDL_version.h"], [root])
            self.assertTrue((root / "android-project/app/src/main/java/org/libsdl/app/NativeApi.java").is_file())
