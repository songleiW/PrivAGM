#Installs dependencies needed on a new AWS instance
sudo apt update
sudo apt install gcc
sudo apt install make
sudo apt install g++

sudo apt-get install automake build-essential git libboost-dev libboost-thread-dev libsodium-dev libssl-dev libtool m4 python texinfo yasm

# wget https://github.com/data61/MP-SPDZ/archive/refs/tags/v0.3.2.zip
# unzip v0.3.2.zip
# cd MP-SPDZ-0.3.2

git clone --recursive --branch v0.3.2 https://github.com/data61/MP-SPDZ.git
cd MP-SPDZ
mkdir Player-Data
make tldr
make shamir