import os
from glob import glob

import pandas as pd

from src.config import cfg


def make_df(path_list):
    df_all = pd.DataFrame()
    for path in path_list:
        with open(path, "r", encoding='UTF-8') as f:
            lines = f.readlines()

        Q = []
        A = []

        for i in range(len(lines)):
            # TODO: 예외 처리 항목 추가 => Q, A의 개수가 안맞을 때 처리 프로세스 추가
            try:
                if i % 2 == 0:
                    Q.append(lines[i][4:-1])
                    A.append(lines[i + 1][4:-1])
            except:
                pass
        df = pd.DataFrame()
        df['Q'] = Q
        df['A'] = A
        df['label'] = 1

        df_all = pd.concat([df_all, df])

    df_all.to_csv(os.path.join(cfg.DATA_PATH, "QnA_data.csv"), index=False)


if __name__ == "__main__":
    data_path = glob(cfg.DATA_PATH + "/*.txt")
    make_df(data_path)
