import pandas as pd
import numpy as np
import os
import shutil

def save_duplicates(fname, mt_pairs):
	with open(fname, 'w') as f:
		for i,elem in enumerate(mt_pairs):
			f.write('' + str(i) + ': ' + elem[0] + ' - ' + elem[1] + '\n')
			
			
def write_duplicates(mt_pairs):
	with open('tmp.txt', 'w') as f:
		for i,elem in enumerate(mt_pairs):
			f.write('' + str(i) + ': ' + elem[0] + ' - ' + elem[1] + '\n')
	
def make_copy(fname):
	shutil.copyfile('tmp.txt',fname)

def cleanup():
	if os.path.exists('tmp.txt'):
		os.remove('tmp.txt')
	else:
		print('No temporary files to remove')
