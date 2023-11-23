# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from threading import Event, Thread
import numpy as np
import psutil
import time
import os

def sample_next_token(logits: np.ndarray, top_k=20, top_p=0.7, temperature=1):
    # softmax with temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits / temperature)
    probs = exp_logits / np.sum(exp_logits)

    # top k
    top_k_idx = np.argsort(-probs)[:top_k]
    top_k_probs = probs[top_k_idx]

    # top p
    cumsum_probs = np.cumsum(top_k_probs)
    top_k_probs[(cumsum_probs - top_k_probs) > top_p] = 0.0
    top_k_probs = top_k_probs / np.sum(top_k_probs)

    # sample
    next_token = np.random.choice(top_k_idx, size=1, p=top_k_probs)
    return next_token[0].item()


def flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def is_gptq(config):
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"


class MemConsumption:
    def __init__(self):
        """Initialize MemConsumption."""
        self.g_exit_get_mem_thread = False
        self.g_end_collect_mem = False
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1
        self.g_event = Event()
        self.g_data_event = Event()

    def collect_memory_consumption(self):
        """Collect the data."""
        while self.g_exit_get_mem_thread is False:
            self.g_event.wait()
            while True:
                process = psutil.Process(os.getpid())
                rss_mem_data = process.memory_info().rss / float(2**20)
                try:
                    shared_mem_data = process.memory_info().shared / float(2**20)
                except Exception:
                    shared_mem_data = -1
                if rss_mem_data > self.g_max_rss_mem_consumption:
                    self.g_max_rss_mem_consumption = rss_mem_data
                if shared_mem_data > self.g_max_shared_mem_consumption:
                    self.g_max_shared_mem_consumption = shared_mem_data
                self.g_data_event.set()
                if self.g_end_collect_mem is True:
                    self.g_event.set()
                    self.g_event.clear()
                    self.g_end_collect_mem = False
                    break
                time.sleep(500 / 1000)

    def start_collect_memory_consumption(self):
        """Start collect."""
        self.g_end_collect_mem = False
        self.g_event.set()

    def end_collect_momory_consumption(self):
        """Stop collect."""
        self.g_end_collect_mem = True
        self.g_event.wait()

    def get_max_memory_consumption(self):
        """Return the data."""
        self.g_data_event.wait()
        self.g_data_event.clear()
        return self.g_max_rss_mem_consumption, self.g_max_shared_mem_consumption

    def clear_max_memory_consumption(self):
        """Clear MemConsumption."""
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1

    def start_collect_mem_consumption_thread(self):
        """Start the thread."""
        self.t_mem_thread = Thread(target=self.collect_memory_consumption)
        self.t_mem_thread.start()

    def end_collect_mem_consumption_thread(self):
        """End the thread."""
        self.g_event.set()
        self.g_data_event.set()
        self.g_end_collect_mem = True
        self.g_exit_get_mem_thread = True
        self.t_mem_thread.join()