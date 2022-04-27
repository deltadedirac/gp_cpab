import confuse, argparse
import dpath.util
import pdb

class configManager():

    def __init__(self, config_path = "configs/setup.yaml"):
        self.config = confuse.Configuration('gp_cpab', __name__)
        self.config.set_file(config_path, base_for_paths=True)

    def parserinfo(self, key):
        return dpath.util.get(self.config.get(), key)


if __name__ == "__main__":
    std = configManager();pdb.set_trace()
    std.parserinfo('*/Window_grid')
