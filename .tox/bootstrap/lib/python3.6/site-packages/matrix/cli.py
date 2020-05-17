from __future__ import print_function

import argparse
from os.path import basename
from os.path import exists
from os.path import join

from jinja2 import Environment
from jinja2 import FileSystemLoader

from . import from_file

try:
    from os.path import samefile
except ImportError:
    from os.path import abspath

    def samefile(a, b):
        return abspath(a) == abspath(b)


def main():
    parser = argparse.ArgumentParser(description='Process matrix configuration and fill templates.')
    parser.add_argument('templates', metavar='TEMPLATE', nargs='+',
                        help='A template to pass the results in.')
    parser.add_argument('-c', '--config', dest='config', metavar='FILE',
                        default='setup.cfg',
                        help='Configuration file (ini-style) to pull matrix conf from. Default: %(default)r')
    parser.add_argument('-s', '--section', dest='section', metavar='SECTION',
                        default='matrix',
                        help='Configuration section to use. Default: %(default)r')
    parser.add_argument('-d', '--destination', dest='destination', metavar='DIRECTORY',
                        default='.',
                        help='Destination of template output. Default: %(default)r')

    args = parser.parse_args()
    jinja = Environment(
        loader=FileSystemLoader('.'),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True
    )
    print('Creating matrix from {0}[{1}] ... '.format(args.config, args.section), end='')
    matrix = from_file(args.config, section=args.section)
    print('DONE.')

    for name in args.templates:
        print('Processing {0} ... '.format(name), end='')
        dest = join(args.destination, basename(name))
        if exists(dest) and samefile(name, dest):
            raise RuntimeError("This would override the template. Use a different destination.")
        with open(dest, "w") as fh:
            fh.write(jinja.get_template(name).render(matrix=matrix))
        print("DONE.")


if __name__ == "__main__":
    main()
