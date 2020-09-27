# image-blending-neon

Image blending using Arm NEON registers. Probably it should work faster than processing single pixels.

Alpha & Beta has constant 0.5 value because program uses shifting bits instead of dividing. Probably some day it will be changed.

Tested on arm64 Debian 10 istalled with Debootstraper. Check "Use Qemu user mode" paragraph for details https://wiki.debian.org/Arm64Qemu.
