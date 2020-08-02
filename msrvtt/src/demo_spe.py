import evaluate_util
from evaluate_util import *
spe = stanford_parsetree_extractor()
input1_parses = spe.run('./input1.txt')
input2_parses = spe.run('./input2.txt')

for i1, i2 in zip(input1_parses, input2_parses):
	print i1
	print i2
	ted = compute_tree_edit_distance(i1, i2)
	print ted