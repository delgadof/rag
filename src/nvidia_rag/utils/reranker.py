# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The wrapper for interacting with reranking models.
1. _get_ranking_model: Creates the ranking model instance.
2. get_ranking_model: Returns the ranking model instance if it doesn't exist in cache.
"""

import logging
from functools import lru_cache
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_nvidia_ai_endpoints import NVIDIARerank

from nvidia_rag.utils.common import get_config, sanitize_nim_url

logger = logging.getLogger(__name__)

@lru_cache
def _get_ranking_model(model="", url="", top_n=4, **kwargs) -> BaseDocumentCompressor:
    """Create the ranking model.
    
    Supports bearer token authentication for custom endpoints.
    Bearer token can be provided via:
    - RANKING_BEARER_TOKEN environment variable (configured in settings.ranking.bearer_token)
    - 'bearer_token' keyword argument

    Returns:
        BaseDocumentCompressor: Base class for document compressors.
    """

    settings = get_config()

    # Sanitize the URL
    url = sanitize_nim_url(url, model, "ranking")

    try:
        if settings.ranking.model_engine == "nvidia-ai-endpoints":
            # Check if bearer token is configured
            bearer_token = settings.ranking.bearer_token or kwargs.get('bearer_token', '')
            
            if url:
                logger.info("Using ranking model hosted at %s", url)
                nvidia_rerank_kwargs = {
                    'base_url': url,
                    'top_n': top_n,
                    'truncate': "END"
                }
                
                if bearer_token:
                    logger.info("Using bearer token authentication for ranking endpoint")
                    nvidia_rerank_kwargs['api_key'] = bearer_token
                
                return NVIDIARerank(**nvidia_rerank_kwargs)

            if model:
                logger.info("Using ranking model %s hosted at api catalog", model)
                nvidia_rerank_kwargs = {
                    'model': model,
                    'top_n': top_n,
                    'truncate': "END"
                }
                
                if bearer_token:
                    logger.info("Using bearer token authentication for ranking API catalog")
                    nvidia_rerank_kwargs['api_key'] = bearer_token
                
                return NVIDIARerank(**nvidia_rerank_kwargs)
        else:
            logger.warning("Unable to find any supported ranking model. Supported engine is nvidia-ai-endpoints.")
    except Exception as e:
        logger.error("An error occurred while initializing ranking_model: %s", e)
    return None


def get_ranking_model(model="", url="", top_n=4, **kwargs) -> BaseDocumentCompressor:
    """Create the ranking model."""
    ranker = _get_ranking_model(model, url, top_n, **kwargs)
    if ranker is None:
        logger.warning("Cached ranking model was None — clearing cache and retrying.")
        _get_ranking_model.cache_clear()
        ranker = _get_ranking_model(model, url, top_n, **kwargs)
    return ranker