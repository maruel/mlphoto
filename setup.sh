#!/bin/bash
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

if [ ! -f venv/bin/activate ]; then
  python3 -m venv venv
fi
source venv/bin/activate

pip install -U pip
pip install -U torch torchvision
pip install -U accelerate diffusers einops insightface onnxruntime peft setuptools transformers
pip install git+https://github.com/TencentARC/PhotoMaker.git
pip freeze > requirements-$(uname).txt
