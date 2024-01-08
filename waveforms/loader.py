import importlib
import io
import logging
import sys

import git

log = logging.getLogger(__name__)


class GitFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path, name='HEAD'):
        self.path = path
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        log.debug("find_spec: fullname=%r, path=%r, target=%r", fullname, path,
                  target)
        loader = GitModuleLoader(self.path, self.name, fullname)
        if loader.blob is not None:
            log.debug("find_spec: module %r found", fullname)
            return importlib.util.spec_from_loader(fullname, loader)
        else:
            log.debug("find_spec: module %r not found", fullname)
            return None


class GitModuleLoader(importlib.abc.FileLoader, importlib.abc.SourceLoader):
    def __init__(self, repo_path, revision, fullname):
        self.repo_path = repo_path
        self.hexsha = git.Repo(self.repo_path).commit(revision).hexsha

        self.blob, self.filepath, self._is_package = self._find_blob(
            repo_path, revision, '/'.join(fullname.split('.')))
        
    def get_blob(self, revision, filepath):
        try:
            # find a module
            blob = git.Repo(self.repo_path).commit(revision).tree[filepath + '.py']
            return blob, filepath + '.py'
        except KeyError:
            try:
                blob = git.Repo(self.repo_path).commit(revision).tree[filepath + '.pyc']
                return blob, filepath + '.pyc'
            except KeyError:
                return None, None

    def _find_blob(self, repo_path, revision, filepath):
        try:
            # find a module
            blob = git.Repo(repo_path).commit(revision).tree[filepath + '.py']
            return blob, filepath + '.py', False
        except KeyError:
            pass
        try:
            # find a package
            blob = git.Repo(repo_path).commit(revision).tree[filepath +
                                                         '/__init__.py']
            return blob, filepath + '/__init__.py', True
        except KeyError:
            pass
        return None, None, False

    def is_package(self, fullname):
        return self._is_package

    def get_filename(self, fullname):
        return f"git://{self.hexsha}@{self.repo_path}|{self.filepath}"

    def get_data(self, path):
        buf = io.BytesIO()
        self.blob.stream_data(buf)
        buf.seek(0)
        return buf.read()


_installed_meta_cache = {}


def install_meta(path, name='HEAD'):
    if (path, name) not in _installed_meta_cache:
        finder = GitFinder(path, name)
        _installed_meta_cache[(path, name)] = finder
        for i, v in enumerate(sys.meta_path):
            if isinstance(v, GitFinder):
                sys.meta_path.insert(i, finder)
                break
        else:
            sys.meta_path.append(finder)
        log.debug('%r installed on sys.meta_path', finder)


def remove_meta(path, name='HEAD'):
    if (path, name) in _installed_meta_cache:
        finder = _installed_meta_cache.pop((path, name))
        sys.meta_path.remove(finder)
        log.debug('%r removed from sys.meta_path', finder)
