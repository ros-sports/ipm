# Copyright (c) 2022 Hamburg Bit-Bots
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

from sensor_msgs.msg import CameraInfo


def sanity_check(camera_info: CameraInfo) -> bool:
    """
    Sanity check if the camera info is valid.
    """
    # Check K intrinsic camera matrix
    k = camera_info.k
    if (k[0] == 0 or k[1] != 0 or k[2] == 0 or
        k[3] != 0 or k[4] == 0 or k[5] == 0 or
        k[6] != 0 or k[7] != 0 or k[8] != 1):
        return False
    return True
