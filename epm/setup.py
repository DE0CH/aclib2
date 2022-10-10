import setuptools
from pathlib import Path
requirements_file = Path(__file__).parent / 'epm/requirements.txt'

setuptools.setup(name="epm",
                 description="Empirical Performance Model",
                 version="0.2",
                 packages=setuptools.find_packages(
                         exclude=["*.tests*", "*.unittests.*", "unittests.*",
                                  "unittests", "tests"
                                  ]),
                 install_requires=requirements_file.read_text().split('\n'),
                 test_suite="nose.collector",
#                scripts=['scripts/surrogate/flask_server.py',
#                         'scripts/surrogate/flask_worker.py',
#                         'scripts/surrogate/g_unicorn_app.py'
#                         ],
                 package_data={'': ['*.txt', '*.md']},
                 author="Katharina Eggensperger, Marius Lindauer"
                        " and Philipp Müller",
                 author_email="{eggenspk,lindauer}@cs.uni-freiburg.de",
                 license="GPLv2",
                 platforms=['Linux'],
                 classifiers=[],
                 url="https://bitbucket.org/mlindauer/epm")
