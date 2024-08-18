from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='guardgraph',
      version='0.0.1',
      description='GUARDIAN graph package',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/AgentschapPlantentuinMeise/guardgraph',
      author='Christophe Van Neste',
      author_email='christophe.vanneste@plantentuinmeise.be',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: POSIX",
          "Development Status :: 1 - Planning"
      ],
      install_requires=[
          'flask',
          'Flask-IAM',
          'neo4j',
          'pandas',
          'requests',
          'matplotlib',
          'tqdm',
          'scikit-learn',
          'imblearn',
          'joblib', # kaggle pickled models
          'owlready2',
          'OSMPythonTools',
          'pygbif',
          'folium',
          'pyproj',
          'celery[sqlalchemy]'
      ],
      extras_require={
          'documentation': ['Sphinx']
      },
      package_data={},
      include_package_data=True,
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'guardgraph-init=guardgraph.__main__:main'
          ],
      },
      test_suite='nose.collector',
      tests_require=['nose']
      )

# To install with symlink, so that changes are immediately available:
# pip install -e .
