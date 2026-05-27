from importlib import resources

def get(filename):
    return resources.files(__package__).joinpath(filename).read_text()
