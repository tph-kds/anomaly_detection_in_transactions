import os
from pathlib import Path

package_name = "ano_detection"

list_of_files = [
    # Root folder
    ".github/workflows/ci_cd.yaml",
    ".github/workflows/python_publish.yaml",
    ".github/workflows/release.yaml",
    "src/__init__.py",
    f"src/{package_name}/__init__.py",
    # Data folder
    f"src/{package_name}/data/__init__.py",

    # Config folder
    f"src/{package_name}/config/__init__.py",

    # Components folder
    f"src/{package_name}/components/__init__.py",
    
    # Exception folder
    f"src/{package_name}/exception/__init__.py",

    # Logger folder
    f"src/{package_name}/logger/__init__.py",

    # Models folder
    f"src/{package_name}/models/__init__.py",

    # Utils folder
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/utils/config.py",
    f"src/{package_name}/utils/logger.py",
    f"src/{package_name}/utils/exception.py",
    
    # Pipeline folder
    f"src/{package_name}/pipeline/__init__.py",
    f"src/{package_name}/tracking/__init__.py",

    # Visualization folder
    f"src/{package_name}/visualization/__init__.py",

    # API folder
    f"src/{package_name}/api/__init__.py",

    # Main folder 
    # Inference folder
    "serving/app.py",
    "serving/k8s/deployment.yaml",
    "serving/k8s/service.yaml",
    "serving/minikube/deployment.yaml",
    "serving/minikube/service.yaml",
    
    # Tests folder
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",

    # Docs folder
    "docs/main.md",

    # readme Info
    "readme/images/"


    # Init folder
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "Dockerfile",
    "docker-compose.yaml",
    "Makefile",
    ".env",
    "dvc.yaml",
    # Experiments folder
    "notebooks/experiments_01.ipynb",

]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass # Create a empty file