# Copyright 2020 Google LLC
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

# Downloads relevant parts of the SVM dataset into folders
# inside the current directory

BASEDIR=$(dirname "$0")
local_data_directory=$BASEDIR/data
mkdir $BASEDIR/data

# To download the binary problems:
data_source="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
for filename in real-sim rcv1_train.binary;
do
    curl $data_source/$filename.bz2 --output $local_data_directory/$filename.bz2
    bunzip2 -d $local_data_directory/$filename.bz2
done

# To download the regression problem:
data_source="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression"

filename=E2006.train
curl $data_source/$filename.bz2 --output $local_data_directory/$filename.bz2
bunzip2 -d $local_data_directory/$filename.bz2
