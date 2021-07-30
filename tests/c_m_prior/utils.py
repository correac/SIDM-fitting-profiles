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
    required=False,
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
    "-w",
    "--wvel",
    help="Initial guess for velocity in cross section model. Default 0.",
    type=float,
    required=False,
    default=None,
)

parser.add_argument(
    "-v",
    "--variable",
    help="Variable: inner slope for velocity dispersion profile. Default 0.",
    type=float,
    required=False,
    default=None,
)

parser.add_argument(
    "-rho",
    "--density_model",
    help="Running with isothermal or modified-isothermal model for density profile? Default modified-isothermal.",
    type=str,
    required=False,
    default=None,
)

parser.add_argument(
    "-sigma",
    "--cross_section_model",
    help="Running with constant or velocity-dependent model for the cross section? Default velocity-dependent.",
    type=str,
    required=False,
    default=None,
)

parser.add_argument(
    "-sample",
    "--halo_sample",
    help="Running for joint profile or individual halos profile?. Default joint.",
    type=str,
    required=False,
    default=None,
)

args = parser.parse_args()

if args.halo_sample is None:
    args.halo_sample = "joint"

if args.cross_section_model is None:
    args.cross_section_model = "velocity-dependent"

if args.density_model is None:
    args.density_model = "modified-isothermal"

if args.wvel is None:
    args.wvel = 0

if args.variable is None:
    args.variable = 0




