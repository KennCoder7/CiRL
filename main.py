#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
import torch
from torchlight import import_class
# from tools.seed_everything import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processor collection')
    # seed_everything(3)

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['recognition_cirl'] = import_class('processor.recognition_cirl.REC_Processor')

    #endregion yapf: enable


    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    arg.device = torch.cuda.current_device()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    p.start()
