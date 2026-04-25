# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Math Escalation Environment - A pure MCP environment for math escalation.
"""

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .math_client import MathEnv

__all__ = ["MathEnv", "CallToolAction", "ListToolsAction"]
