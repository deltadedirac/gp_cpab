import confuse, argparse
import dpath.util
import pdb
import re


class configManager():

    def __init__(self, config_path = "../configs/setup.yaml"):
        self.config = confuse.Configuration('gp_cpab', __name__)
        self.regex = re.compile('^\*\/')
        if isinstance(config_path, argparse.Namespace):
            self.config.set_args(config_path)
        else:
            self.config.set_file(config_path, base_for_paths=True)

    def parserinfo(self, key):
        return dpath.util.get(self.config.get(), key)

    def get_config_vals(self,keys):
        if filter(self.regex.match,keys):
            final = [ self.parserinfo(k) for k in keys ]
            return final
        else:
            return [ self.config.get()[k] for k in keys ]


if __name__ == "__main__":
    std = configManager()
    std.parserinfo('*/Window_grid')
