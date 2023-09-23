import os
import shutil
from pathlib import Path
from typing import List, Optional


class OsProcessor:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def get_all_files(self, directory_path: Optional[str] = None) -> List[str]:
        """Returns a list of all files in a directory"""
        if directory_path is None:
            directory_path = self.directory_path

        all_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                all_files.append(os.path.join(root, file))

        return all_files

    def get_files_with_extension(
        self, extension: str, directory_path: Optional[str] = None
    ) -> List[str]:
        """Returns a list of all files in a directory with a given extension"""
        if directory_path is None:
            directory_path = self.directory_path

        all_files = self.get_all_files(directory_path)
        files_with_extension = [file for file in all_files if file.endswith(extension)]

        return files_with_extension

    def get_file_extension(self, file_path: str) -> str:
        """Returns the extension of a file"""
        return Path(file_path).suffix

    def get_subdirectories(self, directory_path: Optional[str] = None) -> List[str]:
        """Returns a list of all subdirectories in a directory"""
        if directory_path is None:
            directory_path = self.directory_path

        subdirectories = [
            os.path.join(directory_path, d)
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d))
        ]

        return subdirectories

    def create_directory(self, directory_path: str) -> None:
        """Creates a directory if it does not exist"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def delete_directory(self, directory_path: str) -> None:
        """Deletes a directory if it exists"""
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)

    def copy_file(self, source_path: str, destination_path: str) -> None:
        """Copies a file from one location to another"""
        shutil.copy2(source_path, destination_path)

    def move_file(self, source_path: str, destination_path: str) -> None:
        """Moves a file from one location to another"""
        shutil.move(source_path, destination_path)
