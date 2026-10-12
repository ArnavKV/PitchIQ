"""Verify the export boundary instead of relying on research .gitignore rules."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from check_publication import check_files
from export_public import export_public


class PublicationTests(unittest.TestCase):
    def test_export_has_no_data_models_or_history_and_manifest_matches(self):
        with TemporaryDirectory() as temporary:
            directory, archive, _ = export_public(ROOT, Path(temporary) / "portfolio")
            paths = [path.relative_to(directory) for path in directory.rglob("*") if path.is_file()]
            self.assertEqual(check_files(directory, paths), [])
            self.assertFalse(any(set(path.parts) & {"data", "models", ".git", ".venv", "archive"} for path in paths))
            self.assertNotIn(Path("docs/preview.png"), paths)
            manifest = json.loads((directory / "PUBLIC_MANIFEST.json").read_text())
            for relative, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((directory / relative).read_bytes()).hexdigest(), digest)
            with zipfile.ZipFile(archive) as package:
                self.assertEqual(len(package.namelist()), len(paths))
                self.assertTrue(all(name.startswith("portfolio/") for name in package.namelist()))

    def test_existing_export_is_preserved(self):
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "portfolio"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("existing work")
            with self.assertRaises(FileExistsError):
                export_public(ROOT, destination)
            self.assertEqual(marker.read_text(), "existing work")

    def test_unreviewed_and_credential_material_is_rejected(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "data").mkdir()
            (directory / "data/matches.csv").write_text("external dataset")
            (directory / "README.md").write_text("-" * 5 + "BEGIN " + "PRIVATE KEY" + "-" * 5)
            issues = check_files(directory, [Path("data/matches.csv"), Path("README.md")])
            self.assertEqual(len(issues), 2)
            self.assertTrue(any("possible private key" in reason for _, reason in issues))
            self.assertTrue(all("BEGIN" not in reason for _, reason in issues))


if __name__ == "__main__":
    unittest.main()
