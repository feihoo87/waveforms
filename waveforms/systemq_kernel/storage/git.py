import hashlib
from pathlib import Path

import git


def track_repository(name: str, url: str,
                     path: Path) -> tuple[git.Repo, git.Repo, str]:
    """
    Track a repository by name and url.

    If the repository is already tracked, return the existing repository.
    Otherwise, clone the repository and return the clone.

    :param name: The name of the repository.
    :param url: The url of the repository.
    :param path: The path to the directory where the repository should be stored.
    :return: A tuple of the origin repository, the tracked repository, and the url.
    """
    origin = git.Repo(url)
    md5 = hashlib.md5(str(url).encode('utf-8')).hexdigest()
    path = path / f'{name}_{md5}.git'
    if path.exists() and path.is_dir():
        repo = git.Repo(path)
    else:
        repo = git.Repo.init(path=path, bare=True)
        remote = repo.create_remote(name='origin', url=url)
        remote.fetch()
    return origin, repo, str(url)


def get_heads_of_repositories(
    repositories: dict[str, tuple[git.Repo, git.Repo, str]]
) -> dict[str, tuple[str, str]]:
    """
    Get the heads of the repositories.

    :param repositories: A dictionary of the repositories.
    :return: A dictionary of the heads of the repositories.
    """
    ret = {}
    for name, (origin, repo, url) in repositories.items():
        ret[name] = origin.commit('HEAD').hexsha, url
        try:
            repo.commit(ret[name][0])
        except:
            repo.remote("origin").fetch()
    return ret
