"""
Class implementing a simple file browser for Jupyter
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3


import ipywidgets as ipw
from pathlib import Path


class Folders:
    def __init__(self):

        self.cur_dir = Path(".").resolve()
        self.out = ipw.Output()

        self.file_list = ipw.Select(rows=10, layout={"width": "500px"})
        self.file_list.options = [".."] + self.get_files()
        self.file_list.value = None
        self.file_list.observe(self.move_folders, names="value")

        self.refresh_button = ipw.Button(description="Refresh")
        self.refresh_button.on_click(self.refresh)

    def get_files(self):

        current_files = [x.name for x in self.cur_dir.glob("*") if not x.is_dir()]
        current_folders = [x.name for x in self.cur_dir.glob("*") if x.is_dir()]
        current_files = sorted(current_files, key=str.lower)
        current_folders = sorted(current_folders, key=str.lower)

        return current_folders + current_files

    def refresh(self, b):

        self.file_list.unobserve(self.move_folders, names="value")
        self.file_list.options = [".."] + self.get_files()
        self.file_list.value = None
        self.file_list.observe(self.move_folders, names="value")

    def move_folders(self, change):
        if change["new"] == "..":

            self.cur_dir = self.cur_dir.resolve().parent

            self.file_list.unobserve(self.move_folders, names="value")
            self.file_list.options = [".."] + self.get_files()
            self.file_list.value = None
            self.file_list.observe(self.move_folders, names="value")

        elif change["new"] is not None:

            self.cur_dir = self.cur_dir.joinpath(change["new"])
            if self.cur_dir.is_dir():

                self.file_list.unobserve(self.move_folders, names="value")
                self.file_list.options = [".."] + self.get_files()
                self.file_list.value = None
                self.file_list.observe(self.move_folders, names="value")

            else:
                self.cur_dir = self.cur_dir.resolve().parent

