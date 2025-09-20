"""Utilities for downloading the Citi Bike SQLite dataset from Kaggle.

This module mirrors the workflow from ``collab/db_load.ipynb`` and exposes
functions that make it easy to reuse the dataset setup from other notebooks or
scripts. The main entry point is :func:`prepare_citibike_database`, which
returns both the SQLite connection object and a convenience query function.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd

# Default configuration mirrors the notebook values.
DEFAULT_DATASET_SLUG = "benhamner/sf-bay-area-bike-share"
DEFAULT_SQLITE_FILENAME = "database.sqlite"
DEFAULT_DATA_DIR = Path("citibike_data")
KAGGLE_TOKEN_PATH = Path.home() / ".kaggle" / "kaggle.json"


class KaggleSetupError(RuntimeError):
    """Raised when the Kaggle CLI is not ready for use."""


def ensure_kaggle_credentials(token_path: Path = KAGGLE_TOKEN_PATH) -> Path:
    """Ensure the Kaggle API token is available locally.

    Parameters
    ----------
    token_path:
        Location of the Kaggle API token. Defaults to ``~/.kaggle/kaggle.json``.

    Returns
    -------
    Path
        The resolved path to the Kaggle token file.

    Raises
    ------
    FileNotFoundError
        If the token file is missing. The caller should prompt the user to add
        the token before attempting to download datasets.
    """

    token_path = token_path.expanduser()
    if token_path.exists():
        os.chmod(token_path, 0o600)
        return token_path

    # Attempt to prompt the user for a token when running in Google Colab.
    try:
        from google.colab import files  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise FileNotFoundError(
            "Kaggle API token not found. Upload kaggle.json manually or place it at "
            f"{token_path}."
        ) from exc

    print("Kaggle API token not found. Please upload your kaggle.json file.")
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError(
            "No file uploaded. Run the cell again and select kaggle.json."
        )

    token_bytes = uploaded.get("kaggle.json")
    if token_bytes is None:
        token_bytes = next(
            (data for name, data in uploaded.items() if name.lower() == "kaggle.json"),
            None,
        )
    if token_bytes is None:
        raise FileNotFoundError(
            "Uploaded files did not include kaggle.json. Try again with the correct file."
        )

    token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    token_path.write_bytes(token_bytes)
    os.chmod(token_path, 0o600)
    print(f"Saved kaggle.json to {token_path}")
    return token_path


def ensure_kaggle_cli_available() -> None:
    """Check that the Kaggle CLI is installed and on PATH."""

    if shutil.which("kaggle") is None:
        raise KaggleSetupError(
            "The 'kaggle' command-line tool is not available. Install it with "
            "'pip install kaggle' and ensure it is on your PATH."
        )


def download_sqlite_dataset(
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    sqlite_filename: str = DEFAULT_SQLITE_FILENAME,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    force_download: bool = False,
) -> Path:
    """Download the specified Kaggle dataset and extract the SQLite file.

    Parameters
    ----------
    dataset_slug:
        The Kaggle dataset slug ``<owner>/<dataset>``.
    sqlite_filename:
        The filename to extract from the dataset archive.
    data_dir:
        Destination directory for the downloaded archive and extracted database.
    force_download:
        If ``True`` forces a fresh download even when the SQLite file exists.

    Returns
    -------
    Path
        Path to the extracted SQLite database on disk.
    """

    ensure_kaggle_cli_available()
    ensure_kaggle_credentials()

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / sqlite_filename

    if sqlite_path.exists() and not force_download:
        return sqlite_path

    archive_name = dataset_slug.split("/")[-1] + ".zip"
    archive_path = data_dir / archive_name

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(data_dir),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise KaggleSetupError(
            "Failed to download dataset from Kaggle. Double-check your token "
            "and internet connectivity."
        ) from exc

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Expected archive {archive_path} after download, but it was not found."
        )

    with zipfile.ZipFile(archive_path) as zf:
        if sqlite_filename not in zf.namelist():
            raise FileNotFoundError(
                f"{sqlite_filename} not found in the Kaggle archive {archive_path.name}."
            )
        zf.extract(sqlite_filename, path=data_dir)

    # Remove the archive to conserve space.
    try:
        archive_path.unlink()
    except FileNotFoundError:
        pass

    return sqlite_path


def connect_to_sqlite(sqlite_path: Path | str) -> sqlite3.Connection:
    """Create a connection to the SQLite database with sensible defaults."""

    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}.")

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def _build_query_runner(conn: sqlite3.Connection) -> Callable[[str, Optional[tuple]], pd.DataFrame]:
    """Create a convenience function that executes read-only queries."""

    def run_query(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        return pd.read_sql_query(sql, conn, params=params or ())

    return run_query


def prepare_citibike_database(
    *,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    sqlite_filename: str = DEFAULT_SQLITE_FILENAME,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    force_download: bool = False,
    build_query_runner: bool = True,
) -> Tuple[Path, sqlite3.Connection, Optional[Callable[[str, Optional[tuple]], pd.DataFrame]]]:
    """Ensure the Citi Bike dataset is ready and return useful database handles.

    Returns a tuple ``(sqlite_path, connection, query_runner)`` where
    ``query_runner`` may be ``None`` when ``build_query_runner`` is ``False``.
    The tuple can be unpacked by downstream notebooks, e.g.::

        sqlite_path, conn, run_query = prepare_citibike_database()

    The ``conn`` object can be provided directly to LangChain utilities that
    expect a database connection, while ``run_query`` mirrors the helper used in
    the original notebook.
    """

    sqlite_path = download_sqlite_dataset(
        dataset_slug=dataset_slug,
        sqlite_filename=sqlite_filename,
        data_dir=data_dir,
        force_download=force_download,
    )
    conn = connect_to_sqlite(sqlite_path)
    query_runner = _build_query_runner(conn) if build_query_runner else None
    return sqlite_path, conn, query_runner


__all__ = [
    "KaggleSetupError",
    "prepare_citibike_database",
    "download_sqlite_dataset",
    "connect_to_sqlite",
    "ensure_kaggle_credentials",
    "ensure_kaggle_cli_available",
]


if __name__ == "__main__":
    path, connection, run_query = prepare_citibike_database()
    print(f"SQLite database ready at {path}")
    tables = run_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;") # type: ignore
    print(tables.head())
