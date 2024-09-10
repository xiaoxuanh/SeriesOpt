import os

# Get the directory of the current file (__file__ refers to the utils.py file)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one level to get the SeriesOpt directory
SERIESOPT_DIR = os.path.dirname(CURRENT_DIR)

# The Data folder is parallel to SeriesOpt, so move up one more level and then locate Data
BASE_DIR = os.path.dirname(SERIESOPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "Data")
SOLVER_DIR = os.path.join(BASE_DIR, "Solvers")

def get_data_path(filename=""):
    """
    Function to get the full path for a data file in the Data directory.
    """
    return os.path.join(DATA_DIR, filename)

def get_solver_path(solver_name):
    """
    Function to get the full path for a solver executable.
    """
    return os.path.join(SOLVER_DIR, solver_name)

def get_results_path(filename=""):
    """
    Function to get the full path for a results file in the Results directory.
    """
    return os.path.join(CURRENT_DIR, "tests", filename)