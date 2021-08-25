# Copyright 2021 The FirstOrderLp Authors
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

# This awk script processes a fixed-format MPS file, replacing embedded spaces
# in the fixed-format name fields with "_", and empty names with a single "_".
# This appears to be sufficient to convert netlib fixed-format MPS files
# into ones that can be read by free-format MPS readers, but is not a fully
# general fix, since it doesn't ensure that the conversion doesn't create name
# conflicts (that is, for example, if there was a field with the name "x y" and
# another with the name "x_y", this conversion will cause them to have the same
# name).

# Replaces embedded " " in s with "_". If s is all " ", changes the first " " to
# "_".
function fixname(s,n,t) {
  n = match(s, / *$/);
  if (n == 1) n = 2;
  t=substr(s, 1, n-1);
  gsub(/ /, "_", t);
  return t substr(s, n);
}
# Applies fixname to columns a through b of s, if present.
function fixfield(s,a,b) {
  if (length(s) <= a || match(substr(s, a), " ") == 0) {
    return s;
  }
  return substr(s, 1, a-1) fixname(substr(s, a, b-a+1)) substr(s, b+1);
}
{
  if (substr($0, 1, 1) != " ") {
    print $0;
  } else {
    s = fixfield($0, 5, 12);
    s = fixfield(s, 15, 22);
    s = fixfield(s, 40, 47);
    print s;
  }
}
