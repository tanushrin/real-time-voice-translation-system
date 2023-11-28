from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='VoiceTranslationAPI',
      version="1.0",
      description="Simple FastAPI with voice translation",
      packages=find_packages(),
      include_package_data=True,
      install_requires=requirements)