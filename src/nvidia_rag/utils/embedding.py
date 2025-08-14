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

"""The wrapper for interacting with embedding models.
1. get_embedding_model: Get the embedding model. Uses the NVIDIA AI Endpoints or HuggingFace.
"""

import logging
from functools import lru_cache
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from nvidia_rag.utils.common import get_config, sanitize_nim_url

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:
    logger.warning("Optional module torch not installed.")

@lru_cache
def get_embedding_model(model: str, url: str, **kwargs) -> Embeddings:
    """Create the embedding model.
    
    Supports bearer token authentication for custom endpoints.
    Bearer token can be provided via:
    - EMBEDDING_BEARER_TOKEN environment variable (configured in settings.embeddings.bearer_token)
    - 'bearer_token' keyword argument
    """
    model_kwargs = {"device": "cpu"}
    if torch.cuda.is_available():
        model_kwargs["device"] = "cuda:0"

    encode_kwargs = {"normalize_embeddings": False}
    settings = get_config()

    # Sanitize the URL
    url = sanitize_nim_url(url, model, "embedding")

    logger.info("Using %s as model engine and %s and model for embeddings",
                settings.embeddings.model_engine,
                model)
    if settings.embeddings.model_engine == "huggingface":
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embeddings.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        # Load in a specific embedding model
        return hf_embeddings

    if settings.embeddings.model_engine == "nvidia-ai-endpoints":
        # Check if bearer token is configured
        bearer_token = settings.embeddings.bearer_token or kwargs.get('bearer_token', '')
        
        if url:
            logger.info("Using embedding model %s hosted at %s", model, url)
            nvidia_embeddings_kwargs = {
                'base_url': url,
                'model': model,
                'truncate': "END"
            }
            
            if bearer_token:
                logger.info("Using bearer token authentication for embedding endpoint")
                nvidia_embeddings_kwargs['api_key'] = bearer_token
            
            return NVIDIAEmbeddings(**nvidia_embeddings_kwargs)

        logger.info("Using embedding model %s hosted at api catalog", model)
        nvidia_embeddings_kwargs = {
            'model': model,
            'truncate': "END"
        }
        
        if bearer_token:
            logger.info("Using bearer token authentication for embedding API catalog")
            nvidia_embeddings_kwargs['api_key'] = bearer_token
        
        return NVIDIAEmbeddings(**nvidia_embeddings_kwargs)

    raise RuntimeError(
        "Unable to find any supported embedding model. Supported engine is huggingface and nvidia-ai-endpoints.")
