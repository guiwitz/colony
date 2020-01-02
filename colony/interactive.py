"""
Class implementing a GUI for the analysis of bacterial colonies.
"""
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3


import glob, os, pickle
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

        self.folder_field = ipw.Textarea(
            # description="Folder", value="Enter folder", layout={"width": "700px"}
            description="Folder",
            value="Imagefolder",
            layout={"width": "700px"},
        )

        self.folder_sel = Folders()
        # self.folder_sel._show_dialog()

        self.out = ipw.Output()

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
            description="Analyze selected jpg", style=style, layout={"width": "250px"}
        )
        self.run_button.on_click(self.analyse_single)

        self.runall_button = ipw.Button(
            description="Analyze all jpg", style=style, layout={"width": "250px"}
        )
        self.runall_button.on_click(self.analyse_all)

        self.plot_button = ipw.Button(description="Plot selected jpg",layout={"width": "250px"})
        self.plot_button.on_click(self.plot_result)

        self.save_button = ipw.Button(description="Save results")
        self.save_button.on_click(self.save_results)
        
        self.load_button = ipw.Button(description="Load results")
        self.load_button.on_click(self.load_results)

        self.folder_sel.file_list.observe(self.on_update_folder, names="value")

    def analyse_single(self, b):

        self.run_button.description = "Currently analyzing..."
        for file in self.select_file.value:
            contour, peaks, area = co.complete_analysis(
                self.folder_field.value + "/" + file
            )
            self.results[file] = {"contour": contour, "peaks": peaks, "area": area}
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
            contour, peaks, area = co.complete_analysis(
                self.folder_field.value + "/" + file
            )
            self.results[file] = {"contour": contour, "peaks": peaks, "area": area}
        self.runall_button.description = "Analyze all jpg"

    def plot_result(self, b):

        with self.out:
            clear_output()
            current_file = self.select_file.value[0]
            image = skimage.io.imread(self.folder_field.value + "/" + current_file)

            fig, ax = plt.subplots(figsize=(10, 10))
            plt.imshow(image, cmap="gray")
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

            fig.tight_layout()
            plt.show()

    def on_update_folder(self, change):

        self.files = [
            os.path.split(x)[1] for x in self.folder_sel.cur_dir.glob("*.jpg")
        ] + [os.path.split(x)[1] for x in self.folder_sel.cur_dir.glob("*.JPG")]
        self.select_file.options = tuple(self.files)
        
    def save_results(self, b):

        if not os.path.isdir(self.folder_field.value + "/Result"):
            os.makedirs(self.folder_field.value + "/Result")
        file_to_save = self.folder_field.value + "/Result/results.pkl"
        with open(file_to_save, "wb") as f:
            pickle.dump(self.results, f)

        pd.DataFrame(
            [
                {
                    "filename": x,
                    "num_peaks": len(self.results[x]["peaks"]),
                    "area": self.results[x]["area"],
                }
                for x in self.results
            ]
        ).to_csv(self.folder_field.value + "/Result/summary.csv", index=False)
        
    def load_results(self, b):
        
        file_to_load = self.folder_field.value + "/Result/results.pkl"
        
        with open(file_to_load, 'rb') as f:
            self.results = pickle.load(f)