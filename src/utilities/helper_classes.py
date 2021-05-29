from shutil import move
import pickle
import sys
import os

#### Create save directories and copy the main script

class save_dir():
    
    def __init__(self, args, basedir, testcase_dir, save_user_inputs=True):
        self.args = args
        self.basedir = basedir
        self.testcase_dir = testcase_dir
        self.save_user_inputs = save_user_inputs
        
        os.chdir(self.basedir)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        self.checkpoint_dir = os.path.join(self.args.model_dir, "ckpt")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "cp-{epoch:04d}.ckpt")
        if not os.path.exists(self.checkpoint_dir):
          os.makedirs(self.checkpoint_dir)

        self.figsave_dir = os.path.join(self.args.model_dir, "img")
        if not os.path.exists(self.figsave_dir):
            os.makedirs(self.figsave_dir)

        
    def __call__(self, script_name):
        os.chdir(self.basedir)

        os.system("jupyter nbconvert --to python " + os.path.join(self.testcase_dir, script_name + ".ipynb"))
        move(os.path.join(self.testcase_dir, script_name + ".py"), os.path.join(self.args.model_dir, "orig_run_file.py"))

        if self.save_user_inputs:
            with open(os.path.join(self.args.model_dir, 'args.pkl'), 'wb') as output:
                pickle.dump(self.args, output, pickle.HIGHEST_PROTOCOL)
