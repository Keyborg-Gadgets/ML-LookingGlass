# Real time looking glass for Windows


<<<<<<< HEAD
This project is not quite ready for release. There's a few flaws which I'll cover in caveats. I'm applying for some roles around development tooling so I'm showcasing some of that.


The long term goal of this project is a thesis on how we communicate in code and disseminate technical information. Its intentionally less than DRY. I want every idea to be tangible. It's not a "shader", its math on a "texture", but it's just an image of whats on your screen. Architecturally how can we communicate technology boundaries in a way that can be quickly understood and referenced. This repo is not just code, it's a conversation.


Conway's Law says: organizations which design systems are constrained to produce designs which are
copies of the communication structures of these organizations


The the thesis succinctly is, can we flip conways law on it's head? Can we design systems and communicate in code, in such a way we spawn new organizations and structures? I believe so.


https://github.com/user-attachments/assets/29991ccc-9300-40b7-9b9f-c8cac50b3b93

# Architecture 

![Blank diagram(2)](https://github.com/user-attachments/assets/eb9dc0d2-b5d1-466b-877a-566ee63a5387)


# Building


I still need to add the onnx builder to the code, so at this time we go onnx->engine with trtexec. All of this is handled for you.


Requirements:
```
Visual Studio 2022 with Cmake (this will be gone soon)
Windows SDK 10.0.26100.0 (this will be gone soon)
Python (this will be gone soon)
```


```
git clone https://github.com/Keyborg-Gadgets/ML-LookingGlass.git
cd ML-LookingGlass/LookingGlass/Dependency-Generator
pip install playwright
./launch.cmd
cd ..
.\BuildCuda.ps1
```


It's going to open a browser window so you can auth for the TensorRT Deps. [Nothing I can do](https://github.com/NVIDIA/TensorRT/issues/697). It will take some time to build the engine. From there open the project in visual studio and everything should just build and work, crazy right?


# Caveats
Can’t handle DPI scaling. I find the window by scanning for an ICON which changes if you scale the DPI. I’ll fix this.


Can’t handle most Laptops because they usually pass through the integrated CPU. You have to output via the HDMI port.


Tested on single monitor, YMMV.
=======
This project is not quite ready for relase. There's a few flaws which I'll cover in caveats. I'm applying for some roles around development tooling so I'm showcasing some of that. 

https://github.com/Keyborg-Gadgets/ML-LookingGlass/blob/master/Assets/glass.mp4

[![Now in Android: 55]
(https://i.ytimg.com/vi/Hc79sDi3f0U/maxresdefault.jpg)]
(https://www.youtube.com/watch?v=Hc79sDi3f0U "Now in Android: 55")
>>>>>>> bc27699 (Readme)
