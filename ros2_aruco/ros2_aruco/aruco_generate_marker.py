"""
Script for generating Aruco marker images.

Author: Nathan Sprague
Version: 10/26/2020
"""

import argparse
import cv2
import numpy as np


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    """ Trick to allow both defaults and nice formatting in the help. """
    pass


def main():
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter,
                                     description="Generate a .png image of a specified maker.")
    parser.add_argument('--id', default=None, type=int,
                        help='Single marker id to generate (if not using --range)')
    parser.add_argument('--range', nargs=2, type=int, metavar=('START_ID', 'END_ID'),
                        help='Generate a range of marker ids inclusive (e.g., 0 49)')
    parser.add_argument('--size', default=200, type=int,
                        help='Side length in pixels')
    parser.add_argument('--outdir', default='.', type=str,
                        help='Output directory for generated images')
    dict_options = [s for s in dir(cv2.aruco) if s.startswith("DICT")]
    option_str = ", ".join(dict_options)
    dict_help = "Dictionary to use. Valid options include: {}".format(option_str)
    parser.add_argument('--dictionary', default="DICT_4X4_50", type=str,
                        choices=dict_options,
                        help=dict_help, metavar='')
    args = parser.parse_args()

    dictionary_id = cv2.aruco.__getattribute__(args.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    def write_marker(marker_id: int):
        image = np.zeros((args.size, args.size), dtype=np.uint8)
        dictionary.generateImageMarker(marker_id, args.size, image, 1)
        path = f"{args.outdir.rstrip('/')}/marker_{marker_id:04d}.png"
        cv2.imwrite(path, image)

    if args.range is not None:
        start_id, end_id = args.range
        if end_id < start_id:
            start_id, end_id = end_id, start_id
        for marker_id in range(start_id, end_id + 1):
            write_marker(marker_id)
    else:
        if args.id is None:
            raise SystemExit("Either --id or --range START END must be provided")
        write_marker(args.id)


if __name__ == "__main__":
    main()
