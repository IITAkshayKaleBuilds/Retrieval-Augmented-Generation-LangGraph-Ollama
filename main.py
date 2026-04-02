from config import setup_env
setup_env()

from RAG Applications.scripts.my_tools import Test_my_tools_file
from RAG Applications.scripts.nodes import Test_nodes_file
from RAG Applications.scripts.utils import Test_utils_file
from RAG Applications.scripts.init import Test_init_file

print("Running main file & test functions for RAG Applications...")

if __name__ == "__main__":
    Test_my_tools_file()
    Test_nodes_file()
    Test_utils_file()
    Test_init_file
    print("All test functions executed successfully.")
