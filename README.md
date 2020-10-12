# image-blending-neon

Image adding(blending) using Arm NEON registers. Probably it should work faster than processing single pixels.

Program uses OpenCV, so make sure it's installed. Compiling is very hard: ```make```. Check makefile for details;

Usage: ```./lab2 image_name1 image_name2 alpha```

I was working on arm64 Ubuntu(or Debian, I'm not sure) emulator istalled with Debootstraper. Check "Use Qemu user mode" paragraph for details https://wiki.debian.org/Arm64Qemu. Also tested on Jetson Nano.
