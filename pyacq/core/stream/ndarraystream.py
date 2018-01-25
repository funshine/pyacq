# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

import struct
import zmq
import numpy as np

from .streamhelpers import DataSender, DataReceiver, register_transfermode
from .arraytools import make_dtype

class NdarrayDataSender(DataSender):
    """Helper class to send data serialized over socket.
    
    Note: this class is usually not instantiated directly; use
    ``OutputStream.configure(transfermode='ndarray')``.
    
    To avoid unnecessary copies (and thus optimize transmission speed), data is
    sent exactly as it appears in memory including array strides.
    
    """
    def send(self, index, data):
        # optional pre-processing before send
        if isinstance(data, np.ndarray):
            for f in self.funcs:
                index, data = f(index, data)
                
        """send a numpy array with metadata"""
        md = dict(
            dtype = str(data.dtype),
            shape = data.shape,
            index = index,
        )
        flags = 0
        copy = self.params.get('copy', True)
        self.socket.send_json(md, flags|zmq.SNDMORE)
        self.socket.send(data, flags, copy=copy)


class NdarrayDataReceiver(DataReceiver):
    """Helper class to receive data serialized over socket.
    
    See NdarrayDataSender.
    """
    def __init__(self, socket, params):
        DataReceiver.__init__(self, socket, params)
    
    def recv(self, return_data=True):
        """recv a numpy array"""
        flags = 0
        md = self.socket.recv_json(flags=flags)
        msg = self.socket.recv(flags=flags)
        # buf = buffer(msg)
        # convert to array
        dtype = make_dtype(self.params['dtype']) # this avoid some bugs but is not efficient because this is call every sends...
        # dtype = self.params['dtype']     
        data = np.frombuffer(msg, dtype=md['dtype'])
        index = md['index']
        data = data.reshape(md['shape'])

        if not return_data:
            return index, None

        return index, data


register_transfermode('ndarray', NdarrayDataSender, NdarrayDataReceiver)
