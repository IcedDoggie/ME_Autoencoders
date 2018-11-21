
import argparse

def simple_func():
	print("nothing in here")

FUNCTION_MAP = {'top20' : simple_func,
}

parser.add_argument('command', choices=FUNCTION_MAP.keys())

args = parser.parse_args()

func = FUNCTION_MAP[args.command]
func()