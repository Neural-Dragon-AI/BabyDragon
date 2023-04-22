from github import Github


class IssueParser:
    def __init__(self, repo_name):
        self.g = Github()
        self.repo = self.g.get_repo(repo_name)

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


class CommitParser:
    def __init__(self, repo_name):
        self.g = Github()
        self.repo = self.g.get_repo(repo_name)

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
