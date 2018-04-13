#!/usr/bin/env python3


import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description='Feature Builder')
    parser.add_argument('stances_dataset', metavar='stances_dataset',
                        help='Stances dataset.')
    parser.add_argument('bodies_dataset', metavar='bodies_dataset',
                        help='Bodies dataset.')
    parser.add_argument('output_file', metavar='output_file',
                        help='Output file.')
    args = parser.parse_args()

    stances = pd.read_csv(args.stances_dataset)
    bodies = pd.read_csv(args.bodies_dataset)

    print(stances.head())
    print(bodies.head())
    print(stances.merge(bodies, on="Body ID"))


if __name__ == '__main__':
    main()
