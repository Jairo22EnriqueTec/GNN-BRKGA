import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-n", "--Negro", help = "", type=int)
parser.add_argument("-c", "--Chupa", help = "", type=str)
args = parser.parse_args()
print(args.Negro)
print(args.Chupa)