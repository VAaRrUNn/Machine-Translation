import setuptools
from typing import List
with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

__version__ = "0.0.0"
REPO_NAME = "Machine-Translation"
AUTHOR_USER_NAME = "VaruN-dev-dev"
SRC_REPO = "machineTranslation"
AUTHOR_EMAIL = "sanatoo.varun666@gmail.com"



HYPEN_E = '-e .'
def get_requirements(file_path: str) -> List:
    """
    This function will return the list of requirements, 
    and ignore the -e .
    """
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", '') for req in requirements]

        if HYPEN_E in requirements:
            requirements.remove(HYPEN_E)
    return requirements


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="A small python package for English to Hindi translation",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    # project_urls =
    package_dir={"": "machineTranslation"},
    packages=setuptools.find_packages(where="machineTranslation"),
    install_requires = get_requirements("requirements.txt"),
)
