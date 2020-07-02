from setuptools import setup, find_packages

setup(name='gkmodel',
      version='0.0.0',
      description='Goal-keeper model',
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'matplotlib',
                        'xspline'],
      zip_safe=False)
