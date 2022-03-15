import os
import configparser


class MyParser(configparser.ConfigParser):
    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


def get_config(CONFIG_FILE_PATH):
    cfg = MyParser()
    cfg.read(CONFIG_FILE_PATH)
    return cfg.as_dict()


if __name__ == '__main__':
    cfg = MyParser()
    from EIVideo import CONFIG_FILE_PATH
    cfg.read(CONFIG_FILE_PATH)
    print(cfg.as_dict())
