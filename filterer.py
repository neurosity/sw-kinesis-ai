import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn.base import BaseEstimator, TransformerMixin

class Filterer(TransformerMixin):
    def __init__(self,
                 metrics_labels=None,
                 filter_order=5,
                 filter_high=60.,
                 filter_low=0.5,
                 filter_window=15,
                 nb_chan=8,
                 sample_rate=250.0,
                 signal_buffer_length=2500):
        self.metrics_labels = metrics_labels
        self.count_ = 0
        self.tiny_count_ = 0
        self.nb_chan = nb_chan
        self.sample_rate = sample_rate
        self.filtered_ = RingBufferSignal(np.zeros((nb_chan + 2, signal_buffer_length)))
        self.signal_buffer_length = signal_buffer_length
        self.filter_high = filter_high
        self.filter_low = filter_low
        self.filter_order = filter_order
        self.filter_window = filter_window
        self.tiny_signal_ = RingBufferSignal(np.zeros((nb_chan + 2, self.filter_window)))
        cofs = butter(filter_order,
                      np.array([filter_low, filter_high]) / (sample_rate / 2.0),
                      'bandpass')
        self.b = cofs[0]
        self.a = cofs[1]
        self.zi_ = lfilter_zi(self.b, self.a)
        self.zf = np.zeros((nb_chan, len(self.zi_)))
        self.zf[:] = self.zi_

    def reset(self):
        """
        Resets the internals of the signal.
        :return:
        """
        self.count_ = 0
        self.zf[:] = self.zi_
        self.filtered_ = RingBufferSignal(np.zeros((self.nb_chan + 2, self.signal_buffer_length)))
        self.tiny_signal_ = RingBufferSignal(np.zeros((self.nb_chan + 2, self.filter_window)))

    def get_epoch(self, signal_size=250):
        return self.filtered_[:-2, -signal_size:]

    def get_cov(self, epoch_time=2500, timestamp=0, signal_transformer=None):
        """Used to get a covariance matrix from the filtered signal buffer

        Parameters
        ----------
        epoch_time : time - ms
            size of the slice you want to build a cov matrix out of. Hacks it
            from the `signalBuffer` which is a `RingBuffer`
        timestamp : time - ms
            The time stamp of the offset of the epoch. 0 by default
        signal_transformer : FlatChannelRemover | None, optional
            A transformer to mix in if needed.
        Returns
        -------
        ndarry : shape(num_chan, num_chan) | shape(num_good_chan, num_good_chan)
        """
        threshold = int(1. / self.sample_rate * 1000.) - 1
        num_samples = int(epoch_time / 1000. * self.sample_rate)
        if timestamp == 0:
            temp_signal = self.filtered_[:-2, -num_samples:]
            if signal_transformer is not None:
                temp_signal = signal_transformer.transform(temp_signal)
            return np.array(np.cov(temp_signal))
        else:
            latest_time = self.get_latest_time()
            if timestamp <= (latest_time + threshold):
                for i in range(1, self.signal_buffer_length):
                    cur = int(self.filtered_[-1, -i])
                    if cur <= timestamp + threshold:
                        temp_signal = self.filtered_[:-2, -(num_samples+i):-i]
                        if signal_transformer is not None:
                            temp_signal = signal_transformer.transform(temp_signal)
                        return np.array(np.cov(temp_signal))
            else:
                raise ValueError('time stamp not in buffer')

    def get_latest_time(self):
        """
        Get the last time in the filtered buffer
        :return: int
            The last time in the buffer. None if there is no data in the buffer.
        """
        return int(self.filtered_[-1, :].max())

    def _filter_with_time(self, X):
        """

        :param X:
        :return:
        """
        filt = lfilter(self.b, self.a, X[:-2, :])
        filtered_with_time = np.zeros(X.shape)
        filtered_with_time[:-2, :] = filt
        filtered_with_time[-2:, :] = X[-2:, :]
        return filtered_with_time

    def _filter_with_time_zf(self, X):
        """

        :param X:
        :return:
        """
        filt, self.zf = lfilter(self.b, self.a, X[:-2, :], zi=self.zf)
        filtered_with_time = np.zeros(X.shape)
        filtered_with_time[:-2, :] = filt
        filtered_with_time[-2:, :] = X[-2:, :]
        return filtered_with_time

    def transform(self, X):
        """
        Filters the incoming signal
        :param X:
        :param y:
        :return:
        """
        return self._filter_with_time(X)

    def partial_transform(self, X):
        """

        :param X:
        :param y:
        :return:
        """
        num_elements = 1 if len(X.shape) != 2 else X.shape[1]
        self.count_ += num_elements

        if num_elements > self.filter_window:
            filtered_with_time = self._filter_with_time_zf(X)
            self.filtered_.extend(filtered_with_time)
            return filtered_with_time

        if num_elements == 1:
            self.tiny_signal_.append(X)
        else:
            self.tiny_signal_.extend(X)

        self.tiny_count_ += num_elements

        if self.tiny_count_ >= self.filter_window:
            filtered_with_time = self._filter_with_time_zf(self.tiny_signal_)
            self.filtered_.extend(filtered_with_time)
            self.tiny_count_ = 0

        return self.filtered_[:, -num_elements:]

class RingBufferSignal(np.ndarray):
    """A multidimensional ring buffer."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def extend(self, xs):
        'Adds array xs to the ring buffer. If xs is longer than the ring '
        'buffer, the last len(ring buffer) of xs are added the ring buffer.'
        xs = np.asarray(xs)
        if self.shape[:1] != xs.shape[:1]:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, xs.shape))
        len_self = self.shape[1:][0]
        len_xs = xs.shape[1:][0]
        if len_self <= len_xs:
            xs = xs[:, -len_self:]
            len_xs = xs.shape[1:][0]
        else:
            self[:, :-len_xs] = self[:, len_xs:]
        self[:, -len_xs:] = xs

    def append(self, x):
        """Adds element x to the ring buffer."""
        x = np.asarray(x)
        self[:, :-1] = self[:, 1:]
        self[:, -1] = x


class RingBuffer(np.ndarray):
    'A multidimensional ring buffer.'

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def extend(self, xs):
        'Adds array xs to the ring buffer. If xs is longer than the ring '
        'buffer, the last len(ring buffer) of xs are added the ring buffer.'
        xs = np.asarray(xs)
        if self.shape[1:] != xs.shape[1:]:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, xs.shape))
        len_self = len(self)
        len_xs = len(xs)
        if len_self <= len_xs:
            xs = xs[-len_self:]
            len_xs = len(xs)
        else:
            self[:-len_xs] = self[len_xs:]
        self[-len_xs:] = xs

    def append(self, x):
        'Adds element x to the ring buffer.'
        x = np.asarray(x)
        if self.shape[1:] != x.shape:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, x.shape))
        len_self = len(self)
        self[:-1] = self[1:]
        self[-1] = x