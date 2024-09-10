from setuptools import setup, find_packages

setup(
    name='SeriesOpt',               # Name of your package
    version='0.1',                  # Version number
    description='Optimization package for time-series environments',  
    author='Xiaoxuan Hou',          # Author name 
    author_email='xxhou@uw.edu',    # Author email
    packages=find_packages(),       # Automatically finds all packages in the directory
    install_requires=[              # External dependencies your package requires
        'numpy',
        'pandas',
        'matplotlib',
    ],
    python_requires='>=3.6',        # Minimum Python version required
)
