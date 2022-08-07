# Copyright (c) 2022 Kenji Brameld
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

from ipm_library.sanity_camera_info import sanity_check
from sensor_msgs.msg import CameraInfo


def test_sanity_check():

    # Check K intrinsic camera matrix is in format:
    #     [fx  0 cx]
    # K = [ 0 fy cy]
    #     [ 0  0  1]
    assert sanity_check(CameraInfo(k=[1, 0, 1, 0, 1, 1, 0, 0, 1]))
    assert not sanity_check(CameraInfo(k=[0, 0, 1, 0, 1, 1, 0, 0, 1]))
    assert not sanity_check(CameraInfo(k=[1, 1, 1, 0, 1, 1, 0, 0, 1]))
    assert not sanity_check(CameraInfo(k=[1, 0, 0, 0, 1, 1, 0, 0, 1]))
    assert not sanity_check(CameraInfo(k=[1, 0, 1, 1, 1, 1, 0, 0, 1]))
    assert not sanity_check(CameraInfo(k=[1, 0, 1, 0, 0, 1, 0, 0, 1]))
    assert not sanity_check(CameraInfo(k=[1, 0, 1, 0, 1, 0, 0, 0, 1]))
    assert not sanity_check(CameraInfo(k=[1, 0, 1, 0, 1, 1, 1, 0, 1]))
    assert not sanity_check(CameraInfo(k=[1, 0, 1, 0, 1, 1, 0, 1, 1]))
    assert not sanity_check(CameraInfo(k=[1, 0, 1, 0, 1, 1, 0, 0, 0]))
