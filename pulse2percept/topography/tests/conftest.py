import pytest

def pytest_addoption(parser):
    try:
        parser.addoption(
            "--runslow", action="store_true", default=False, help="run slow tests"
        )
    except:
        # already added it
        pass

def pytest_configure(config):
    try:
        config.addinivalue_line("markers", "slow: mark test as slow to run")
    except:
        pass


def pytest_collection_modifyitems(config, items):
    try:
        if config.getoption("--runslow"):
            # --runslow given in cli: do not skip slow tests
            return
    except:
        pass
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)