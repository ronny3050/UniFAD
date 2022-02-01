"""Main testing file for universal attack detection using 4-Chimneys
"""
# MIT License
# 
# Copyright (c) 2022 Debayan Deb
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse, utils
from nntools.common.dataset import Dataset
from nntools.common.imageprocessing import preprocess
from nntools.tensorflow.networks import ChimneyCNN

def main(args):
    config = utils.import_file(args.model_path + "/config.py", 'config')

    network = ChimneyCNN()
    network.load_model(args.model_path)

    dataset = Dataset('data/examples')
    proc_func = lambda images: preprocess(images, config, False)
    test_images = proc_func(dataset.images)
    outputs = network.extract_feature(test_images)
    print(outputs)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="The path to the saved checkpoint and model file",
                        type=str)
    args = parser.parse_args()
    main(args)