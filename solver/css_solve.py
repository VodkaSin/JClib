import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import utils.ops as ops
import qutip
import qutip.piqs as piqs
import numpy as np
import time
import matplotlib.pyplot as plt

theta = 