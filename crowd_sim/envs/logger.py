import numpy as np
import torch


class Logger:
    def __init__(self, logger_size: int) -> None:
        self._idx = 0  # 次にデータを挿入するインデックス
        self._size = 0  # データ数
        self._logger_size = logger_size  # リプレイバッファのサイズ
        self._log_data: list[dict] = [{} for _ in range(logger_size)]
        assert logger_size > 0

    def append(self, log_datas: dict) -> None:
        self._log_data[self._idx] = log_datas
        self._idx = (self._idx + 1) % self._logger_size
        self._size = min(self._size + 1, self._logger_size)

    def get_data_old_order(self, max_num: int) -> list[dict]:
        """
        最新のデータからn個取って、古い順に並べた絵データを返す
        """
        num = min(max_num, self._size)
        assert num >= 0
        idl: int = self._idx - num  # [idl, _idx)を取り出す
        if idl < 0:
            idl += self._logger_size
            return self._log_data[idl:] + self._log_data[:self._idx]
        else:
            return self._log_data[idl:self._idx]  # [idl, _idx)

    def get_all_data(self) -> list[dict]:
        return self.get_data_old_order(self._size)

    def clear(self) -> None:
        self._idx = 0
        self._size = 0
        self._log_data: list[dict] = [{} for _ in range(self._logger_size)]


if __name__ == '__main__':
    # test codes for logger
    logger = Logger(5)
    for i in range(5):
        logger.append({"a": i, "b": i})
        print(logger.get_data_old_order(i + 1))
    print("check1: done")

    nums = np.random.randint(0, 6, (10,))
    for i, n in enumerate(nums):
        logger.append({"a": i + 5, "b": i + 5})
        tmp = logger.get_data_old_order(n)
        print(len(tmp), n, tmp)
    print("check2: done")
