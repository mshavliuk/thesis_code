import os

from pyspark import daemon, worker


def remote_debug_wrapped(*args, **kwargs):
    import pydevd_pycharm
    DEBUG_HOST = os.environ['DEBUG_HOST']
    DEBUG_PORT = int(os.environ['DEBUG_PORT'])
    pydevd_pycharm.settrace(host=DEBUG_HOST,
                            port=DEBUG_PORT,
                            stdoutToServer=True,
                            stderrToServer=True,
                            suspend=False)
    print(f'Connected to remote debugger on {DEBUG_HOST}:{DEBUG_PORT}')
    worker.main(*args, **kwargs)


if __name__ == '__main__':
    daemon.worker_main = remote_debug_wrapped
    daemon.manager()
