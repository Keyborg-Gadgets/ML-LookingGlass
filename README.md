# Real time looking glass for Windows

The long term goal of this project is a thesis on how we communicate in code and disseminate technical information. Its intentionally less than DRY. I want every idea to be tangible. It's not a "shader", its math on a "texture", but it's just an image of whats on your screen. Architecturally how can we communicate technology boundaries in a way that can be quickly understood and referenced. This repo is not just code, it's a conversation.


Conway's Law says: organizations which design systems are constrained to produce designs which are
copies of the communication structures of these organizations


The thesis succinctly is, can we flip conways law on it's head? Can we design systems and communicate in code, in such a way we spawn new organizations and structures? I believe so.


https://github.com/user-attachments/assets/29991ccc-9300-40b7-9b9f-c8cac50b3b93

## Runtime Requirements
Any 2 Series GPU and up should work. It takes time to compile the model for your system. It will compile the engine on launch and then not need to compile again.  

# Architecture 

![Blank diagram(2)](https://github.com/user-attachments/assets/eb9dc0d2-b5d1-466b-877a-566ee63a5387)


# Building
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

# Todo
Publish tooling for trimming paddle paddle models.

Cleanup (so much cleanup). But the globals are intentional. This is a playground. It's going to continue to be more of a notebook. I don't know if I like the globals being this global. They'll all get moved to their own headers. But remain public. Drop in, hack anywhere. That's the goal. Grab the header, drop it anywhere. 

# Caveats
Runs at, at least 144FPS on a 3090ti. You can only detect against rendered textures so your monitor has to go that fast.

Only tested on single GPU system. 3090ti and 4090 mobile.

Can’t handle DPI scaling. I find the window by scanning for an ICON which changes if you scale the DPI. I’ll fix this.

Can’t handle most Laptops because they usually pass through the integrated CPU. You have to output via the HDMI port.

Tested on single monitor, YMMV.

# License
Im using [ORB](https://github.com/Accustomer/CUDA-ORB/tree/main). It has no license idk what that means. All my code respects licensing. I never pulled from a repo that wasn't MIT licensed, to my knowledge, except fot ORB. Ill do accreditations when this is all complete.

I don't even have a copyright notice on this. LLMs pose a unique challenge. If you use LLMs, you know what they can do, but you know what they can't do. This code has been modified regurgitated taken from other repos passed through ChatGPT back to me back to ChatGPT back to me. This is still my own fundamentally unique creation. I think this is an incredibly strong software architecture  implementation. But I cant tell you what I created and what someone else created. Many of the algorithms and structures are uniquely mine.
