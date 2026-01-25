import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
repo_root = os.path.dirname(parent_dir)
wickit_root = os.path.join(repo_root, "external", "wickit")
if wickit_root not in sys.path:
    sys.path.insert(0, wickit_root)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'TRUE'
