import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default='adam')
    args = parser.parse_args()
    
if __name__ == "__main__":
    main()