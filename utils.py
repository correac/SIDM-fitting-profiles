"""
Some helper functions .

Contains the argument parser and default parsing results.

See the README for available argument parsers.
"""

import argparse as ap
#import glob
#from typing import Optional

parser = ap.ArgumentParser(
    description="""General argument parser for SIDM-Fitting scripts."""
)

parser.add_argument(
    "-i",
    "--input",
    help="File containing the density profiles to analyse. Required.",
    type=str,
    required=True,
)

parser.add_argument(
    "-o",
    "--output",
    help="Output files directory.",
    type=str,
    required=True,
    default=None,
)

parser.add_argument(
    "-n",
    "--name",
    help="Output file names",
    type=str,
    required=True,
    default=None,
)

parser.add_argument(
    "-s",
    "--sigma",
    help="Initial guess for cross section",
    type=int,
    required=True,
    default=None,
)

args = parser.parse_args()


