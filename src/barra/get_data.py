"""
@author Jacob
@time 2019/06/05
"""

from qi.utils.constants import barra_loader
import sys
from os.path import dirname
import click


@click.command()
@click.option('--end', default="20190604", help='end date str, default 20190604', type=str)
def main(end: str):
    start = '19990129'

    fr = barra_loader.load_factor_return(from_=start, to_=end)
    fr = fr.pivot('date', 'factor', 'return').reset_index()

    fr.to_csv(dirname(__file__) + r'\data\factor_return.csv', index=False)

    print(fr.shape)
    print(fr.head())


if __name__ == '__main__':
    main(sys.argv[1:])
