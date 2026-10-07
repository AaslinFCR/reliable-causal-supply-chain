"""Package source, aggregate results and figures without redistributing datasets."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "reliable-causal-supply-chain.zip"
    allowed_roots = {
        "scrc",
        "tests",
        "configs",
        "scripts",
        "reports",
        "experiments",
        "runs",
        "web",
        "artifacts",
        ".github",
    }
    allowed_top = {
        "README.md",
        "pyproject.toml",
        "requirements-lock.txt",
        "run.py",
        "run.ps1",
        "Makefile",
        ".gitignore",
        ".gitattributes",
        "DEPLOYMENT.md",
        "BUSINESS_SETUP.md",
        "requirements-runtime.txt",
        "Dockerfile",
        ".dockerignore",
        ".env.example",
        "compose.yaml",
        "render.yaml",
        "start-app.ps1",
    }
    excluded = {
        "source_inspection.txt",
        "probe_prices.json",
        "probe_quantities.json",
        "official_probe.json",
    }
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if (
                relative.parts[0] not in allowed_roots
                and str(relative) not in allowed_top
            ):
                continue
            if (
                "__pycache__" in relative.parts
                or path.suffix in {".pyc", ".parquet"}
                or path.name in excluded
                or path.name.startswith("idp_")
            ):
                continue
            archive.write(
                path, arcname="reliable-causal-supply-chain/" + relative.as_posix()
            )
    print(f"{output}: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
