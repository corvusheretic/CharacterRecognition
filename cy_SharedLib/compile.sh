#!/bin/bash
if [ $1 == "clean" ]; then
	cd ./features
	make clean -f cyMakeMB
	cd ../hmm
	make clean -f cyMakeHMM
	cd ..
	make clean -f arMake 
	rm -rf build
	mkdir build
fi

if [ $1 == "exe" ]; then
	cd ./features
	make -f arMakeMB
	cd ../hmm
	make -f arMakeHMM
	cd ..
	make -f arMake
fi

if [ $1 == "so" ]; then
	rm -rf build
	mkdir build
	cd ./features
	make -f cyMakeMB
	cp ../build/cyMartiBunke.so /home/kalyan/Eclipse/Python/CharacterHMM/api/features/.
	cd ../hmm
	make -f cyMakeHMM
	cp ../build/cyBakis.so /home/kalyan/Eclipse/Python/CharacterHMM/api/base/.
	cp ../build/cyBaumWelch.so /home/kalyan/Eclipse/Python/CharacterHMM/api/base/.
	cp ../build/cyCalcBackward.so /home/kalyan/Eclipse/Python/CharacterHMM/api/base/.
	cp ../build/cyCalcForward.so /home/kalyan/Eclipse/Python/CharacterHMM/api/base/.
fi
