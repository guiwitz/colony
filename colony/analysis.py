"""
Class implementing a GUI for the analysis of bacterial colonies.
"""
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3


import glob, os, pickle, re
import colony.colonies as co
import ipywidgets as ipw
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output

from .folders import Folders


class Analysis:
    def __init__(self):

        style = {"description_width": "initial"}

        self.folder_sel = Folders()
        #self.folder_sel.file_list.observe(self.on_update_folder, names="value")
        
        self.out = ipw.Output()
        
        self.repl_select = ipw.Select(options = [])
        self.days_select = ipw.SelectMultiple(options = [])
        self.exp_select = ipw.Select(options = [])
        self.exp_select.observe(self.update_replicas_days, names = 'value')
        
        self.days_select.observe(self.plotting, names = 'value')
        
        
        self.load_button = ipw.Button(description = 'Load experiment')
        self.load_button.on_click(self.on_update_folder)
        
        self.save_button = ipw.Button(description = 'Save plot')
        self.save_button.on_click(self.save_plot)
        
        self.outplot = ipw.Output()
        
    def on_update_folder(self, change):
        '''Load analysis files for all folders present in current folder'''
    
        folder_list = list(self.folder_sel.cur_dir.glob('*'))
        folder_list = [x for x in folder_list if x.is_dir()]
        folder_list = [x for x in folder_list if not re.search('\..*',x.stem)]
        
        
        experiment = pd.concat([pd.DataFrame(pd.read_pickle(x.joinpath('Result/results.pkl').as_posix())).T for x in folder_list])
        experiment = experiment.reset_index()
        experiment = experiment.rename(columns = {'index':'filename'})
        
        experiment['exp_name'] = experiment.filename.apply(lambda x : re.findall('.*?_(.*)?-\d_d\d.*',x)[0])
        experiment['repl_ind'] = experiment.filename.apply(lambda x : int(re.findall('.*-(\d)_d\d.*',x)[0]))
        experiment['day'] = experiment.filename.apply(lambda x : int(re.findall('.*-\d_d(\d).*',x)[0]))
        
        self.experiment = experiment
        self.exp_select.options = np.unique(experiment['exp_name'])
        
        
        #self.exp_grouped = experiment.groupby(['exp_name','repl_ind'])
        
        #self.repl_select.options = list(self.exp_grouped.indices.keys())
        
    def update_replicas_days(self,change):
        '''When picking an experiment, update available replicas and days'''
            
        subsel = self.experiment[self.experiment.exp_name == self.exp_select.value]
        self.days_select.options = np.unique(subsel['day'])
        self.repl_select.options = np.unique(subsel['repl_ind'])
        
        
    def plotting(self, change = None):
        '''Plot colony contour for current selection'''
    
        subsel = self.experiment[self.experiment.exp_name == self.exp_select.value]
        subsel= subsel[subsel.day.isin(self.days_select.value)]
        subsel = subsel[subsel.repl_ind == self.repl_select.value]
        #subsel = subsel[subsel.day]
        
        with self.out:
            clear_output()
            fig, ax = plt.subplots(figsize = (10,10))

            for x in subsel.index:
                plt.plot(subsel.loc[x]['contour'][:,1]-subsel.loc[x]['center_mass'][1],
                        subsel.loc[x]['contour'][:,0]-subsel.loc[x]['center_mass'][0],'k')
            ax.set_aspect('equal')#,'datalim')
            #ax.xaxis.set_ticks_position('none') 
            ax.set_xticks([])
            ax.set_yticks([])
            days = ', '.join(map(str,self.days_select.value))
            ax.set_title(self.exp_select.value + '-replica '+str(self.repl_select.value) + ' days '+days)
            fig.set_tight_layout(True)
            plt.show()
            self.fig = fig
            
    def save_plot(self, b = None):
        '''Save current plot'''
    
        days = ''.join(map(str,self.days_select.value))
        self.plotting()
        self.fig.savefig(self.folder_sel.cur_dir.as_posix()+'/'+self.exp_select.value + '_replica_'+str(self.repl_select.value)+'_days'+days+'.png')
        