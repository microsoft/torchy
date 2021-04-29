import os
import lit.formats

config.name = 'Torchy'
config.test_format = lit.formats.TorchyTest()
config.test_source_root = os.path.dirname(__file__)
