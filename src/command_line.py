from typing import List, Tuple, Optional


# Read plot axis limits from command line
def parse_axis_limits(argv: List[str]) -> Tuple[Optional[float], Optional[float]]:

    if '-fixed_axis' in argv:
        arg_pos = argv.index('-fixed_axis')

        try:
            plot_axis_lb = float(argv[arg_pos + 1])
            plot_axis_ub = float(argv[arg_pos + 2])
            return plot_axis_lb, plot_axis_ub
        except IndexError:
            return None, None

    return None, None


# Parse test decoder from command line
def parse_test_decoder(argv) -> Tuple[str, int]:

    try:
        decoder_type = argv[2]
        decoder_num = int(argv[3])
    except IndexError:
        print('argv:', argv)
        print('Example call: python main.py -test_decoder *decoder_type* '
              '*decoder_num*')
        raise

    return decoder_type, decoder_num
