"""
Some helper functions .

Contains the argument parser and default parsing results.

See the README for available argument parsers.
"""

import argparse as ap

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
    "-s",
    "--snapshot",
    help="Snapshot number.",
    type=str,
    required=False,
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
    "-d",
    "--sigma",
    help="Initial guess for cross section",
    type=float,
    required=False,
    default=None,
)


parser.add_argument(
    "-v",
    "--variable",
    help="Variable: inner slope for velocity dispersion profile",
    type=float,
    required=False,
    default=None,
)

parser.add_argument(
    "-hs",
    "--halosample",
    help="Running for joint profile or individual halos profile?. Default joint.",
    type=str,
    required=False,
    default=None,
)

args = parser.parse_args()

if args.halosample is None:
    args.halosample = "joint"





