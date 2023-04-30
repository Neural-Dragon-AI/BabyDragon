import os
import shutil
import subprocess
from typing import List

from github import Github

from babydragon.processors.os_processor import OsProcessor
from babydragon.processors.parsers.python_parser import PythonParser


class GithubProcessor(OsProcessor):
    def __init__(
        self,
        base_directory: str,
        username=None,
        repo_name=None,
        code_parsers=None,
        minify_code: bool = False,
        remove_docstrings: bool = False,
    ):
        self.username = username
        self.repo_name = repo_name
        self.base_directory = base_directory
        self.github = Github()
        self.repo = self.github.get_repo(f"{username}/{repo_name}")
        repo_path = self.clone_repo(self.repo.clone_url)

        OsProcessor.__init__(self, repo_path)
        self.code_parsers = code_parsers or [
            PythonParser(
                repo_path, minify_code=minify_code, remove_docstrings=remove_docstrings
            )
        ]

    def get_public_repos(self):
        """Returns a list of all public repos for the user."""
        user = self.github.get_user(self.username)
        return user.get_repos()

    def clone_repo(self, repo_url: str):
        """Clones the repo at the specified url and returns the path to the repo."""
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        target_directory = os.path.join(self.base_directory, repo_name)

        if os.path.exists(target_directory):
            shutil.rmtree(target_directory)

        subprocess.run(["git", "clone", repo_url, target_directory])

        return target_directory

    def process_repo(self, repo_path=None):
        """Processes the repo at the specified path.
        If no path is specified, the repo at self.directory_path is processed.
        Returns the list of parsed functions and classes."""
        if repo_path is None:
            repo_path = self.directory_path

        for code_parser in self.code_parsers:
            code_parser.directory_path = repo_path
            code_parser.process_directory(repo_path)

    def process_repos(self):
        """Processes all public repos for the user."""
        for repo in self.get_public_repos():
            if not repo.private:
                print(f"Processing repo: {repo.name}")
                repo_path = self.clone_repo(repo.clone_url)
                self.process_repo(repo_path)
                shutil.rmtree(repo_path)

    def get_repo(self, repo_name):
        """Returns the repo with the specified name."""
        user = self.github.get_user(self.username)
        return user.get_repo(repo_name)

    def process_single_repo(self):

        repo = self.get_repo(self.repo_name)
        print(f"Processing repo: {self.repo_name}")
        repo_path = self.clone_repo(repo.clone_url)
        self.process_repo(repo_path)
        shutil.rmtree(repo_path)

    def get_issues(self, state="open"):
        """
        Returns a list of all issues in the repo with the specified state.
        """
        issues = []
        for issue in self.repo.get_issues(state=state):
            issues.append(issue)
        return issues

    def parse_issues(self, state="open"):
        """
        Parses all issues in the repo with the specified state and returns a list of dicts.
        Each dict contains the issue number, title, body, and labels.
        """
        parsed_issues = []
        issues = self.get_issues(state=state)
        for issue in issues:
            parsed_issue = {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "labels": [label.name for label in issue.labels],
            }
            parsed_issues.append(parsed_issue)
        return parsed_issues

    def get_commits(self):
        """
        Returns a list of all commits in the main branch of the repository.
        """
        commits = []
        branch = self.repo.get_branch("main")
        for commit in self.repo.get_commits(sha=branch.commit.sha):
            commits.append(commit)
        return commits

    def parse_commits(self):
        """
        Parses all commits in the main branch of the repository and returns a list of dicts.
        Each dict contains the commit sha, commit message, and author information.
        """
        parsed_commits = []
        commits = self.get_commits()
        for commit in commits:
            parsed_commit = {
                "sha": commit.sha,
                "message": commit.commit.message,
                "author": {
                    "name": commit.commit.author.name,
                    "email": commit.commit.author.email,
                    "date": commit.commit.author.date,
                },
            }
            parsed_commits.append(parsed_commit)
        return parsed_commits
