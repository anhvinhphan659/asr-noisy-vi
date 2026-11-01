import importlib.util
import json
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_code.py"


def _load():
    spec = importlib.util.spec_from_file_location("package_code", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_bundles_code_and_version(tmp_path):
    package_code = _load()
    archive, version = package_code.package(tmp_path / "bundle")
    bundle = tmp_path / "bundle"
    for name in ["core/__init__.py", "dataprep/__init__.py", "evaluation/__init__.py",
                 "pyproject.toml", "README.md", "LICENSE", "CODE_VERSION.json"]:
        assert (bundle / name).is_file(), name
    assert not list(bundle.rglob("__pycache__"))
    assert json.loads((bundle / "CODE_VERSION.json").read_text(encoding="utf-8")) == version
    with zipfile.ZipFile(archive) as z:
        assert "dataprep/pipeline.py" in z.namelist()


def test_bundle_is_installable_layout(tmp_path):
    # The bundle is what Kaggle installs with ``pip install -e``: it must not need tests/.
    package_code = _load()
    package_code.package(tmp_path / "bundle")
    assert not (tmp_path / "bundle" / "tests").exists()
    text = (tmp_path / "bundle" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["core*", "dataprep*", "evaluation*"]' in text
