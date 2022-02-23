import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys 
import logomaker
import pdb

class logomaker_plots():

	
	def __init__(self, alphabet, c2ix, i2cx, i2ix, msa_num):
		self.alphabet = alphabet
		self.c2i = c2ix
		self.i2c = i2cx
		self.i2i = i2ix
		self.msa = msa_num

	#@classmethod
	def integerSeq2strinSeq(self):
		self.msa_str_seq = self.msa.clone().detach()
		#self.msa_str_seq.apply_(lambda x: self.i2c[x.item()])
		tmp = []

		for i in self.msa_str_seq:
			tmp.append( ''.join([self.i2c[ str(j.item()) ] for j in i]) )
		self.msa_str_seq = tmp


	#@classmethod
	def generate_information_matrix(self,i):
		self.integerSeq2strinSeq()
		# create counts matrix
		#domain_counts_df = logomaker.alignment_to_matrix(sequences=self.msa_str_seq,to_type='counts',characters_to_ignore='.-X')
		domain_counts_df = logomaker.alignment_to_matrix(sequences=self.msa_str_seq,to_type='counts')

		# filter base on counts
		num_seqs = domain_counts_df.sum(axis=1)
		pos_to_keep = num_seqs > len(self.msa)/2
		domain_counts_df = domain_counts_df[pos_to_keep]
		domain_counts_df.reset_index(drop=True, inplace=True)

		# transform to information matrix
		domain_info_df = logomaker.transform_matrix(domain_counts_df,
						from_type='counts', to_type='information')
						
		crp_logo = logomaker.Logo(domain_info_df,
                          #shade_below=.5,
                          #fade_below=.5,
                          font_name='Liberation Sans Narrow',
                          #font_name='STIXSizeTwoSym',
                          color_scheme = 'NajafabadiEtAl2017',
                          figsize=(10, 5) )#(120, 5) ) #120,6 Humor Sans
        # style using Logo methods
        #crp_logo.figsize=(30, 80)
		crp_logo.style_spines(visible=False)
		crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
		#crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)
		#pdb.set_trace()
		plt.savefig('logo'+i+'.png')
		plt.show()
