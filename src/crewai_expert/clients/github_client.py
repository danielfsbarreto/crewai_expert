import base64
import os

import aiohttp
import requests


class GithubClient:
    _OWNER = "crewAIInc"
    _REPO = "crewAI"
    _TOKEN = os.getenv("GITHUB_AUTH_KEY")
    _HEADERS = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {_TOKEN}",
    }

    def get_file_paths(self, docs_dir: str) -> list[str]:
        branch_url = f"https://api.github.com/repos/{self._OWNER}/{self._REPO}"

        response = requests.get(branch_url, headers=self._HEADERS)
        if response.status_code != 200:
            raise Exception(f"Failed to get repo info: HTTP {response.status_code}")
        repo_data = response.json()
        default_branch = repo_data["default_branch"]

        commit_url = f"https://api.github.com/repos/{self._OWNER}/{self._REPO}/branches/{default_branch}"
        response = requests.get(commit_url, headers=self._HEADERS)
        if response.status_code != 200:
            raise Exception(f"Failed to get branch info: HTTP {response.status_code}")
        branch_data = response.json()
        commit_sha = branch_data["commit"]["sha"]

        tree_url = f"https://api.github.com/repos/{self._OWNER}/{self._REPO}/git/trees/{commit_sha}?recursive=1"
        response = requests.get(tree_url, headers=self._HEADERS)
        if response.status_code != 200:
            raise Exception(f"Failed to get tree: HTTP {response.status_code}")
        tree_data = response.json()

        return [
            item["path"]
            for item in tree_data["tree"]
            if item["path"].startswith(docs_dir)
            and item["path"].endswith((".mdx", ".md"))
            and item["type"] == "blob"
        ]

    async def get_file_content_async(self, path: str) -> str:
        url = f"https://api.github.com/repos/{self._OWNER}/{self._REPO}/contents/{path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._HEADERS) as response:
                if response.status == 200:
                    data = await response.json()
                    return base64.b64decode(data["content"]).decode("utf-8")
                else:
                    raise Exception(
                        f"Failed to get file content: HTTP {response.status}"
                    )
