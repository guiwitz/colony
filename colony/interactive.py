"""
Class implementing a GUI for the analysis of bacterial colonies.
"""
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3


import glob, os, pickle, re, subprocess
import colony.colonies as co
import ipywidgets as ipw
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output

from .folders import Folders


class Gui:
    def __init__(self):

        style = {"description_width": "initial"}
        layout= {"width": "250px"}

        self.folder_sel = Folders()

        self.out = ipw.Output()
        self.out_check = ipw.Output()
        
        self.files = [
            os.path.split(x)[1] for x in self.folder_sel.cur_dir.glob("*.jpg")
        ] + [os.path.split(x)[1] for x in self.folder_sel.cur_dir.glob("*.JPG")]

        self.select_file = ipw.SelectMultiple(
            options=tuple(self.files),
            # description="Select files to process",
            style=style,
            layout={"width": "350px"},
        )

        self.results = {}

        self.run_button = ipw.Button(
            description="Analyze selected jpg", style=style, layout=layout
        )
        self.run_button.on_click(self.analyse_single)

        self.runall_button = ipw.Button(
            description="Analyze selected folder (single day)", style=style, layout=layout
        )
        self.runall_button.on_click(self.analyse_all)
        
        self.runexp_button = ipw.Button(
            description="Analyze full experiment (multi-folder)", style=style, layout=layout
        )
        self.runexp_button.on_click(self.analyse_experiment)

        self.plot_button = ipw.Button(description="Plot selected jpg",layout=layout)
        self.plot_button.on_click(self.plot_result)

        self.save_button = ipw.Button(description="Save results")
        self.save_button.on_click(self.save_results)
        
        self.load_button = ipw.Button(description="Load results")
        self.load_button.on_click(self.load_results)
        
        self.save_allplots_button = ipw.Button(description="Save all plots")
        self.save_allplots_button.on_click(self.save_all_plots)

        self.folder_sel.file_list.observe(self.on_update_folder, names="value")
        
        self.save_plots_check = ipw.Checkbox(description = 'Save plots during analysis', value = True)
        
        self.zip_button = ipw.Button(description = 'Zip results', style = style, layout = layout)
        self.zip_button.on_click(self.do_zipping)

    def analyse_single(self, b):

        self.run_button.description = "Currently analyzing..."
        for file in self.select_file.value:
            contour, peaks, area, center_mass = co.complete_analysis(
                self.folder_sel.cur_dir.as_posix() + "/" + file
            )
            self.results[file] = {"contour": contour, "peaks": peaks, "area": area, 'center_mass': center_mass}
            
        if self.save_plots_check:
            self.plot_result()
            self.fig.savefig(self.folder_sel.cur_dir.as_posix()+'/Result/'+file+'_seg.png')
            
        self.run_button.description = "Analyze selected jpg"

    def analyse_all(self, b):

        self.runall_button.description = "Currently analyzing..."
        for ind, file in enumerate(self.select_file.options):
            self.runall_button.description = (
                "Currently analyzing "
                + str(ind + 1)
                + "/"
                + str(len(self.select_file.options))
            )
            contour, peaks, area, center_mass = co.complete_analysis(
                self.folder_sel.cur_dir.as_posix() + "/" + file
            )
            self.results[file] = {"contour": contour, "peaks": peaks, "area": area, 'center_mass': center_mass}
            self.save_results(None)
                
            if self.save_plots_check:
                self.select_file.value = (file,)
                self.plot_result()
                self.fig.savefig(self.folder_sel.cur_dir.as_posix()+'/Result/'+file+'_seg.png')
                
        self.runall_button.description = "Analyze all jpg"
        
    def analyse_experiment(self, b):

        temp_cur_dir = self.folder_sel.cur_dir
        folder_list = list(self.folder_sel.cur_dir.glob('*'))
        folder_list = [x for x in folder_list if not re.search('\..*',x.stem)]
        
        for indf, f in enumerate(folder_list):
            
            self.folder_sel.cur_dir = f
            
            self.results = {}
            current_files = [
                os.path.split(x)[1] for x in f.glob("*.jpg")
            ] + [os.path.split(x)[1] for x in f.glob("*.JPG")]
            self.select_file.options = current_files


            self.runall_button.description = "Currently analyzing..."
            for ind, file in enumerate(current_files):
                self.runall_button.description = (
                    "Currently analyzing folder "
                    +str(indf)
                    +' '
                    + str(ind + 1)
                    + "/"
                    + str(len(current_files))
                )
                
                contour, peaks, area, center_mass = co.complete_analysis(
                    self.folder_sel.cur_dir.as_posix() + '/' + file
                )
                self.results[file] = {"contour": contour, "peaks": peaks, "area": area, 'center_mass': center_mass}
                self.save_results(None)
                with self.out:
                    
                    if self.save_plots_check:
                        self.select_file.value = (file,)
                        self.plot_result()
                        self.fig.savefig(self.folder_sel.cur_dir.as_posix()+'/Result/'+file+'_seg.png')

            self.runall_button.description = "Analyze all jpg"
            self.folder_sel.cur_dir = temp_cur_dir

    def plot_result(self, b = None):

        with self.out:
            clear_output()
            current_file = self.select_file.value[0]
            fig, ax = plt.subplots(figsize=(10, 10))
            if self.results[current_file]["contour"] is not None:
                image = skimage.io.imread(self.folder_sel.cur_dir.as_posix() + "/" + current_file)

                
                plt.imshow(image[:,:,1], cmap="gray")
                plt.plot(
                    self.results[current_file]["contour"][:, 1],
                    self.results[current_file]["contour"][:, 0],
                    "r-",
                )
                plt.plot(
                    self.results[current_file]["contour"][
                        self.results[current_file]["peaks"], 1
                    ],
                    self.results[current_file]["contour"][
                        self.results[current_file]["peaks"], 0
                    ],
                    "bo",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            plt.show()
            self.fig = fig

    def on_update_folder(self, change):

        self.files = [
            os.path.split(x)[1] for x in self.folder_sel.cur_dir.glob("*.jpg")
        ] + [os.path.split(x)[1] for x in self.folder_sel.cur_dir.glob("*.JPG")]
        self.select_file.options = tuple(self.files)
        
        #clear results
        self.results = {}
        
    def save_results(self, b=None):

        if not os.path.isdir(self.folder_sel.cur_dir.as_posix() + "/Result"):
            os.makedirs(self.folder_sel.cur_dir.as_posix() + "/Result")
        file_to_save = self.folder_sel.cur_dir.as_posix() + "/Result/results.pkl"
        with open(file_to_save, "wb") as f:
            pickle.dump(self.results, f)

        pd.DataFrame(
            [
                {
                    "filename": x,
                    "num_peaks": len(self.results[x]["peaks"]) if self.results[x]["peaks"] is not None else None,
                    "area": self.results[x]["area"],
                    "center_mass": self.results[x]["center_mass"]
                }
                for x in self.results
            ]
        ).to_csv(self.folder_sel.cur_dir.as_posix() + "/Result/summary.csv", index=False)
        
    def save_all_plots(self, b):
        
        #with self.out_check:
        for f in self.select_file.options:
            self.select_file.value = (f,)
            self.plot_result()
            self.fig.savefig(self.folder_sel.cur_dir.as_posix()+'/Result/'+f+'_seg.png')
            
        
        
    def load_results(self, b):
        
        file_to_load = self.folder_sel.cur_dir.as_posix() + "/Result/results.pkl"
        
        with open(file_to_load, 'rb') as f:
            self.results = pickle.load(f)
            
            
    def do_zipping(self, b):
        """zip the output"""
        
        self.zip_button.description = 'Currently zipping...'
        #save the summary file
        
        with self.out:
            #subprocess.call(['tar', '-czf', 'to_download.tar.gz','-C', self.folders.cur_dir.as_posix(),'.'])
            subprocess.call(['tar', '-czf', self.folder_sel.cur_dir.parent.as_posix()+'/to_download.tar.gz','-C', self.folder_sel.cur_dir.parent.as_posix(), self.folder_sel.cur_dir.name])
        self.zip_button.description = 'Finished zipping!'