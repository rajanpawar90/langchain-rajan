"""Jira Toolkit."""

import jira

class JiraToolkit:
    """A toolkit for interacting with Jira."""

    def __init__(self, url, username, token):
        """Initialize a new JiraToolkit instance.

        Args:
            url (str): The URL of the Jira instance.
            username (str): The username for authentication.
            token (str): The API token for authentication.
        """
        self.url = url
        self.username = username
        self.token = token
        self.jira = jira.Jira(self.url, basic_auth=(self.username, self.token))

    def create_issue(self, summary, description, issue_type='bug'):
        """Create a new Jira issue.

        Args:
            summary (str): The summary of the issue.
            description (str): The description of the issue.
            issue_type (str): The type of issue to create. Default is 'bug'.

        Returns:
            jira.Issue: The created Jira issue.
        """
        return self.jira.create_issue(project='MYPROJECT', summary=summary, description=description, issue_type=issue_type)

    def get_issue(self, issue_id):
        """Get an existing Jira issue.

        Args:
            issue_id (int): The ID of the issue.

        Returns:
            jira.Issue: The Jira issue.
        """
        return self.jira.issue(issue_id)
