#!/usr/bin/env python3
"""FastAPI entry point for LTX-2.3 text-to-video generation."""

import os

from ltx_server_common import ModelProfile, create_server, run_server

MULTI_MODE = int(os.environ.get("LTX_MULTI_MODE", "8"))

PROFILE = ModelProfile(
    display_name=f"LTX-2.3 Video Generator ({MULTI_MODE}x multi)",
    default_width=1024,
    default_height=1024,
    default_frames=121,
    multi_mode=MULTI_MODE,
)

server = create_server(PROFILE)
app = server.app

if __name__ == "__main__":
    run_server(server)
