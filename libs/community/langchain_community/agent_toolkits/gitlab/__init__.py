"""GitLab Toolkit.

This module provides a set of tools for interacting with GitLab.
"""

import requests

class GitLabToolkit:
    """A class representing the GitLab Toolkit.

    Attributes:
        base_url (str): The base URL for the GitLab API.
        private_token (str): The private token for authentication.
    """

    def __init__(self, base_url: str, private_token: str):
        """Initialize the GitLabToolkit object.

        Args:
            base_url (str): The base URL for the GitLab API.
            private_token (str): The private token for authentication.
        """
        self.base_url = base_url
        self.private_token = private_token

    def get_user_info(self) -> dict:
        """Get information about the current user.

        Returns:
            dict: A dictionary containing user information.
        """
        url = f"{self.base_url}/user"
        headers = {"Private-Token": self.private_token}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def create_project(self, name: str, visibility: str = "private") -> dict:
        """Create a new project.

        Args:
            name (str): The name of the project.
            visibility (str, optional): The visibility level of the project.
                Defaults to "private".

        Returns:
            dict: A dictionary containing project information.
        """
        url = f"{self.base_url}/projects"
        data = {"name": name, "visibility": visibility}
        headers = {"Private-Token": self.private_token}
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
