# Realtime looking glass for Windows

This project is not quite ready for release. There's a few flaws which I'll cover in caveats. I'm applying for some roles around development tooling so I'm showcasing some of that.

https://github.com/user-attachments/assets/29991ccc-9300-40b7-9b9f-c8cac50b3b93

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
