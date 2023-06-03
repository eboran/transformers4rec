#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pylint: disable=wildcard-import,unused-import,unused-wildcard-import
import warnings

# Re-export classes/modules from the core library for backwards compatibility
from merlin.io.worker import *  # noqa

warnings.warn(
    "The `nvtabular.worker` module has moved to `merlin.io.worker`. "
    "Support for importing from `nvtabular.worker` is deprecated, "
    "and will be removed in a future version. Please update "
    "your imports to refer to `merlin.io.worker`.",
    DeprecationWarning,
)
