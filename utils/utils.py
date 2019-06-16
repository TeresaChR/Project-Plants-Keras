import argparse


class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

class readable_file(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file=values
        if not os.path.isfile(prospective_file):
            raise argparse.ArgumentTypeError("readable_file:{0} is not a valid path".format(prospective_file))
        if os.access(prospective_file, os.R_OK):
            setattr(namespace,self.dest,prospective_file)
        else:
            raise argparse.ArgumentTypeError("readable_file:{0} is not a readable dir".format(prospective_file))

    
class defaultDirectory(object):
    
    def __init__(self,inputDir, outputDir):
        self.inputDir= inputDir
        self.outputDir= outputDir
        
        
    def get_outputDir(self):
        return self.outputDir
    
    def get_inputDir(self):
        return self.inputDir



def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
   # argparser.add_argument('-i', '--inputDir', action=readable_dir, help='input directory to process')
   # argparser.add_argument('-o', '--outputDir', action=readable_dir, help='output directory to process')
    #argparser.add_argument('-l', '--inputFileList', action=readable_file, help='input file with a list of images')
    
    args = argparser.parse_args()
    
    
    return args
