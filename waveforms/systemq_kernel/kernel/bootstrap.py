import json
from pathlib import Path

from .sched.executor import QuarkExecutor, FakeExecutor
from .sched.sched import bootstrap as _bootstrap

etc = Path.home() / '.systemq'


def bootstrap(config_path: str = etc / 'bootstrap.json'):
    if not config_path.exists():
        config = {
            "executor": {
                "type": "quark",
                "host": "127.0.0.1",
                "port": 2088
            },
            "data": {
                "path": "",
                "url": ""
            },
            "repo": {}
        }
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        with open(config_path) as f:
            config = json.load(f)

    if config['executor']['type'] == 'debug':
        with open(config['executor']['path']) as f:
            executor = FakeExecutor(json.load(f))
    else:
        executor = QuarkExecutor(config['executor']['host'])
    if config['data']['path'] == '':
        datapath = Path.home() / 'data'
        datapath.mkdir(parents=True, exist_ok=True)
    else:
        datapath = Path(config['data']['path'])
    if config['data']['url'] == '':
        url = None
    else:
        url = config['data']['url']
    repo = config.get('repo', None)

    _bootstrap(executor, url, datapath, repo)
