all:
	g++ -O0 lab2.cc -o lab2 -I "/usr/include/opencv4" -L /usr/lib/aarch64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgcodecs
