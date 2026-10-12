"""Check code-only publication boundaries without printing sensitive matches."""
from argparse import ArgumentParser
from pathlib import Path
import re
import subprocess

ROOT_FILES = {
    "app.py", "requirements.txt", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    ".gitignore", ".gitattributes", "PUBLIC_MANIFEST.json",
    "clean_players.py", "load_matches.py", "prepare_players.py", "prepare_shots.py",
    "train_model.py", "train_model_v2.py", "train_model_v3.py", "train_model_v4.py",
    "train_xg.py", "validate_players.py",
}
EXACT_FILES = {
    ".streamlit/config.toml", ".github/workflows/ci.yml",
    "assets/logo.svg", "assets/mark.svg", "assets/style.css", "assets/shot_simulator.html",
    "docs/PUBLICATION.md", "docs/preview-demo.png", "examples/player_data.py",
    "scripts/export_public.py", "scripts/check_publication.py",
}
PYTHON_DIRECTORIES = {"pitchiq", "pages", "views", "tests"}
DENIED_PARTS = {"data", "models", ".venv", "venv", ".aws", ".codex", ".agents", "archive"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "OpenAI key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b"),
}


def allowed(path):
    name = path.as_posix()
    return (
        name in ROOT_FILES or name in EXACT_FILES
        or (len(path.parts) == 2 and path.parts[0] in PYTHON_DIRECTORIES and path.suffix == ".py")
    )


def check_files(directory, paths):
    issues = []
    for relative in paths:
        path = directory / relative
        if relative.is_absolute() or ".." in relative.parts:
            issues.append((relative.as_posix(), "path escapes publication root"))
            continue
        if set(relative.parts) & DENIED_PARTS or not allowed(relative):
            issues.append((relative.as_posix(), "outside reviewed publication allowlist"))
            continue
        if path.is_symlink():
            issues.append((relative.as_posix(), "symbolic links are not allowed"))
            continue
        if not path.is_file():
            issues.append((relative.as_posix(), "missing file"))
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            issues.append((relative.as_posix(), "unexpected file larger than 2 MiB"))
            continue
        if path.suffix.lower() == ".png":
            if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                issues.append((relative.as_posix(), "invalid preview image"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append((relative.as_posix(), "unexpected binary file"))
            continue
        for description, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                issues.append((relative.as_posix(), f"possible {description}; inspect privately"))
    return issues


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--tracked", action="store_true", help="Check git-tracked files, including ignored files already committed")
    parser.add_argument("--verbose", action="store_true", help="Show every finding instead of the first 20")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.tracked:
        output = subprocess.run(["git", "ls-files", "-z"], cwd=directory, capture_output=True, check=True).stdout
        paths = [Path(name.decode()) for name in output.split(b"\0") if name]
    else:
        paths = [path.relative_to(directory) for path in directory.rglob("*") if path.is_file() and not set(path.relative_to(directory).parts) & {".git", "__pycache__", ".venv"}]
    issues = check_files(directory, sorted(paths))
    for path, reason in issues if args.verbose else issues[:20]:
        print(f"BLOCKED {path}: {reason}")
    if issues:
        if not args.verbose and len(issues) > 20:
            print(f"... {len(issues) - 20} additional findings omitted; use --verbose to see them all.")
        print(f"Publication blocked: {len(issues)} issue(s) found in the {'tracked' if args.tracked else 'current'} files.")
        if any(set(Path(path).parts) & {"data", "models", "archive"} for path, _ in issues):
            print("This checkout includes research assets. The portfolio workflow requires the clean source export.")
            print("Publish the contents of dist/pitchiq-public in a NEW repository with fresh Git history.")
            print("Adding .gitignore or deleting current files does not remove their committed history.")
        print("See docs/PUBLICATION.md for the release steps. Keep this check enabled.")
        raise SystemExit(1)
    print(f"Publication file checks passed for {len(paths)} reviewed files. This is not legal certification or a complete secrets audit.")


if __name__ == "__main__":
    main()
