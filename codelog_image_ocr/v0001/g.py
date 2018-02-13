import keras.callbacks

def copy(src, dest):
  with file_io.FileIO(src, mode='rb') as input_f:
    with file_io.FileIO(dest, mode='w+b') as output_f:
        output_f.write(input_f.read())

class Copy(keras.callbacks.Callback):

    def __init__(self, src_fmt, dest_fmt):
        self._src_fmt  = src_fmt
        self._dest_fmt = dest_fmt

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        src  = self._src_fmt.format(epoch=epoch + 1, **logs)
        dest = self._dest_fmt.format(epoch=epoch + 1, **logs)
        copy(src, dest)
