"""
WuYun Brain Simulation - Python Interface
"""
import sys
import os

_LIB_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib', 'Release')
if os.path.isdir(_LIB_DIR):
    sys.path.insert(0, os.path.abspath(_LIB_DIR))

from pywuyun import *
from .viz import plot_raster, plot_connectivity, plot_activity_bars, plot_neuromod_timeline, run_demo
