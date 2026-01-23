# Delete __pycache__ in project dir
# Author: Shengning Wang

import sys
import pathlib
import shutil
from typing import List, Union


def setup_project_root(relative_depth: int = 3) -> pathlib.Path:
    """
    Configures the system path to include the project root directory.

    Args:
        relative_depth (int): The number of levels to traverse up to find root.
            Default is 3 (e.g., from Project/wsnet/utils/del_cache.py to Project/).

    Returns:
        project_root (pathlib.Path): The absolute path to the project root.
    """
    current_file: pathlib.Path = pathlib.Path(__file__).resolve()
    project_root: pathlib.Path = current_file.parents[relative_depth - 1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def clean_python_artifacts(target_dir: Union[str, pathlib.Path] = '.') -> None:
    """
    Recursively removes Python compilation artifacts and cache directories.

    This function targets:
    1. *.pyc and *.pyo files (compiled bytecode).
    2. __pycache__ directories.

    Args:
    - target_dir (Union[str, pathlib.Path]): The root directory to start cleaning.
        Default is the current working directory.

    Note:
    - Uses shutil.rmtree for directory removal to ensure robustness against 
    non-empty cache folders, satisfying industry-standard error handling.
    """
    base_path: pathlib.Path = pathlib.Path(target_dir).resolve()

    # 1. Remove compiled bytecode files
    # Shape: N files matching the glob pattern
    bytecode_patterns: List[str] = ['*.pyc', '*.pyo']
    for pattern in bytecode_patterns:
        for file_path in base_path.rglob(pattern):
            file_path.unlink()

    # 2. Remove cache directories
    # Shape: M directories matching the name '__pycache__'
    for cache_dir in base_path.rglob('__pycache__'):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)


if __name__ == '__main__':
    """
    Main execution logic for environment sanitization and path configuration.
    """
    # Initialize Project Environment
    root: pathlib.Path = setup_project_root(relative_depth=3)

    # Execute Cleanup
    clean_python_artifacts(target_dir='.')
