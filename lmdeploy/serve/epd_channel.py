# Copyright (c) OpenMMLab. All rights reserved.
"""EPD encoder-output transfer payload types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPD_BACKEND_HTTP_JSON = 'http_json'
EPD_BACKEND_DLSLIME_RDMA = 'dlslime_rdma'
EPD_TRANSFER_BACKENDS = (EPD_BACKEND_HTTP_JSON, EPD_BACKEND_DLSLIME_RDMA)


@dataclass
class EncoderTransferEmbedding:
    data: np.ndarray
    start: int
    end: int
    dtype: str
