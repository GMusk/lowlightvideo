import sys
import argparse
import configparser as ConfigParser
from LowLightEnhancer import LowLightEnhancer

""" Use __main__ to run enhancer

usage: __main__.py [-h] -i INPUT -o OUTPUT [-s SIZE SIZE] option

positional arguments:
  option                modus operandi

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input video to enhance.
  -o OUTPUT, --output OUTPUT
                        Path to save enhanced video.
  -s SIZE SIZE, --size SIZE SIZE
                        size of video to display

"""

def required_check(args, arg):
    if arg in args:
        if args[arg] is not None:
            return False
    else:
        return True


def main(argv=None):
    # Do argv default this way, as doing it in the functional
    # declaration sets it at compile time.
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = { "option":"eamc" }

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Defaults")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
        )
    parser.set_defaults(**defaults)
    parser.add_argument('-i', '--input', type=str, required=required_check(defaults, 'input'),
                    help='Path to input video to enhance.')
    parser.add_argument('-o', '--output', required=required_check(defaults, 'output'),
                    help='Path to save enhanced video.')
    parser.add_argument('--option', type=str, help='modus operandi')
    parser.add_argument('-s', '--size', nargs=2, type=int, help='size of video to display')
    parser.add_argument('-b', '--buffer', type=int, help='size of frame buffer')
    args = vars(parser.parse_args(remaining_argv))
    lle = LowLightEnhancer(args)
    lle.enhance()
    return(0)

if __name__ == "__main__":
    main()
