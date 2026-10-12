"""Export reviewed original source without datasets, artifacts, or Git history."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from check_publication import EXACT_FILES, PYTHON_DIRECTORIES, ROOT_FILES, check_files

ROOT = Path(__file__).resolve().parent.parent


def export_public(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}. Choose a new --output directory.")
    archive = destination.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    paths = [Path(name) for name in sorted((ROOT_FILES | EXACT_FILES) - {"PUBLIC_MANIFEST.json"}) if (source / name).is_file()]
    for directory in sorted(PYTHON_DIRECTORIES):
        paths.extend(path.relative_to(source) for path in sorted((source / directory).glob("*.py")))
    required = {"app.py", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "pitchiq/demo.py", "docs/PUBLICATION.md"}
    if missing := required - {path.as_posix() for path in paths}:
        raise ValueError(f"Missing publication inputs: {sorted(missing)}")
    if issues := check_files(source, paths):
        raise ValueError(f"Publication checks found {len(issues)} issue(s): {issues}")
    destination.mkdir(parents=True)
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    manifest = {
        "format": 1,
        "policy": "Original code and fictional demo, with optional runtime CC0 OpenFootball fetch and in-memory model training; no bundled third-party data, models, or prior Git history.",
        "files": {path.as_posix(): hashlib.sha256((destination / path).read_bytes()).hexdigest() for path in sorted(paths)},
    }
    (destination / "PUBLIC_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                package.write(path, Path(destination.name) / path.relative_to(destination))
    return destination, archive, len(paths)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/pitchiq-public")
    args = parser.parse_args()
    directory, archive, count = export_public(ROOT, args.output)
    print(f"Exported {count} reviewed files to {directory}")
    print(f"ZIP: {archive}")
    print("Publish this export in a new repository. The original research history still contains provider datasets.")


if __name__ == "__main__":
    main()
