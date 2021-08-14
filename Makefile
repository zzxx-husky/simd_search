simd: main.cpp
	g++ main.cpp -o simd -mavx2 -g -msse3 -mavx -msse2 -msse4.1

simdr: main.cpp
	g++ main.cpp -o simdr -mavx2 -O3 -msse3 -mavx -msse2 -msse4.1

#simdrm: remove.cpp
#	g++ remove.cpp -o simdrm -mavx2 -g -msse3 -mavx -msse2 -msse4.1
#
#simdrmr: remove.cpp
#	g++ remove.cpp -o simdrmr -mavx2 -O3 -msse3 -mavx -msse2 -msse4.1

all: main.cpp
	make simd
	make simdr
#	make simdrm
#	make simdrmr
