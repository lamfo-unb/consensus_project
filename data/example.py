import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--time', type=float, default=5.0,
                        help='Time in seconds to sleep before operations.')
    parser.add_argument('-s','--symbol',
                        help='CSV file (no header) with the symbols.')
    parser.add_argument('-u', '--username',
                        help='Username deathbycpatcha.')
    parser.add_argument('-p', '--password',
                        help='Pssword deathbycpatcha.')
    parser.add_argument('-l', '--left', type=float, default=230,
                        help='Left position crop captcha.')
    parser.add_argument('-u', '--upper', type=float, default=690,
                        help='Top position crop captcha.')
    parser.add_argument('-r', '--right', type=float, default=630,
                        help='Right position crop captcha.')
    parser.add_argument('-b', '--bottom', type=float, default=800,
                        help='Bottom position crop captcha.')

    #Get arguments
    args = parser.parse_args()

    #Run the data scrapping
    data_scrapping(args)

def data_scrapping(args):
    if args.operation == 'add':
        return args.x + args.y
    elif args.operation == 'sub':
        return args.x - args.y
    elif args.operation == 'mul':
        return args.x * args.y
    elif args.operation == 'div':
        return args.x / args.y


if __name__ == '__main__':
    main()


#python example.py --time=5 --symbol=3 --operation=mul