
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='{{ project_name }}',
      version='0.0.1',
      description='{{ short_description }}',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/AgentschapPlantentuinMeise/{{ project_name }}',
      author='Christophe Van Neste',
      author_email='{{ your_plantentuin_email }}',
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
      ],
      extras_require={
          'documentation': ['Sphinx']
      },
      package_data={},
      include_package_data=True,
      zip_safe=False,
      entry_points={
          'console_scripts': [
              '{{ cli_tool_name}}={{ project_name }}.__main__:main'
          ],
      },
      test_suite='nose.collector',
      tests_require=['nose']
      )

# To install with symlink, so that changes are immediately available:
# pip install -e .
