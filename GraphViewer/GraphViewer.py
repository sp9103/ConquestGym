import matplotlib.pyplot as plt
import csv
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()

    path  = args.path

    pieces = []
    style = ['r^--', 'bs-']
    style_idx = 0
    filenames = os.listdir(path)
    for filename in filenames:
        full_filename = os.path.join(path, filename)
        print(filename)
        pieces.clear()

        with open(full_filename, 'rt') as f:
            data = csv.reader(f, delimiter=',')
            for d in data:
                pieces.append(d)

        pieces.pop(0)
        episode = [Step for WallTime, Step, Value in pieces]
        Value = [Value for WallTime, Step, Value in pieces]

        # 만약 그래프 스무딩이 필요하면 여기서 해야함 -TO-DO

        plt.plot(episode, Value, style[style_idx], label=filename)
        style_idx += 1

    plt.legend()
    plt.title('Compare performance of RL alg')
    plt.xlabel('Episode'), plt.ylabel('Score')
    plt.show()