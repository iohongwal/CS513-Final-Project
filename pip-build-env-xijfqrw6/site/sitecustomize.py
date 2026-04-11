
import os, site, sys

# First, drop system-sites related paths.
original_sys_path = sys.path[:]
known_paths = set()
for path in {'d:\\onedrive - stevens.edu\\26spri cs513 knowledge discovery and data mining\\final project\\cs513-final-project\\.venv\\lib\\site-packages', 'd:\\onedrive - stevens.edu\\26spri cs513 knowledge discovery and data mining\\final project\\cs513-final-project\\.venv'}:
    site.addsitedir(path, known_paths=known_paths)
system_paths = set(
    os.path.normcase(path)
    for path in sys.path[len(original_sys_path):]
)
original_sys_path = [
    path for path in original_sys_path
    if os.path.normcase(path) not in system_paths
]
sys.path = original_sys_path

# Second, add lib directories.
# ensuring .pth file are processed.
for path in ['D:\\OneDrive - stevens.edu\\26Spri CS513 Knowledge Discovery and Data Mining\\Final Project\\CS513-Final-Project\\pip-build-env-xijfqrw6\\overlay\\Lib\\site-packages', 'D:\\OneDrive - stevens.edu\\26Spri CS513 Knowledge Discovery and Data Mining\\Final Project\\CS513-Final-Project\\pip-build-env-xijfqrw6\\normal\\Lib\\site-packages']:
    assert not path in sys.path
    site.addsitedir(path)
