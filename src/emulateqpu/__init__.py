"""Supported finite-model application. Imports perform no numerical work."""
__version__ = '0.1.0'


def run_request(request, output, **options):
    from .api import run_request as run
    return run(request, output, **options)


def example(name='small-pulse'):
    from .contracts import example_request
    return example_request(name)

__all__ = ['__version__', 'run_request', 'example']
