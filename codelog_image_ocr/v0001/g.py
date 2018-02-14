import keras.callbacks
from tensorflow.python.lib.io import file_io
import re
try:
    from google.cloud import storage
except ImportError:
    pass

_client = None
def get_client():
    if _client == None:
        _client = storage.Client()
    return _client

_bucket_dict = {}
def get_bucket(name):
    if name not in _bucket_dict:
        _bucket_dict[name] = get_client().get_bucket(name)
    return _bucket_dict[name]

def parse_url(url):
    if not url.startswith('gs://'):
        return None, None
    m = re.match('gs://([^/]*)/(.*)', url)
    if m is None:
        return None, None
    return m.group(1), m.group(2)

def copy_from_local_to_bucket(src_path, dest_url):
    dest_bucket, dest_path = parse_url(dest_url)
    if dest_bucket is None:
        raise Exception('ZWZYUBEQWW Invalid dest_path: {}'.format(dest_url)')
    blob = get_bucket(dest_bucket).blob(dest_path)
    blob.upload_from_filename(filename=src_path)

class Copy(keras.callbacks.Callback):

    def __init__(self, src_fmt, dest_fmt):
        self._src_fmt  = src_fmt
        self._dest_fmt = dest_fmt

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        src  = self._src_fmt.format(epoch=epoch + 1, **logs)
        dest = self._dest_fmt.format(epoch=epoch + 1, **logs)
        copy_from_local_to_bucket(src, dest)
