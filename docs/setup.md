# Setup

1. Download VirtualHome v2.2.4 executable
   [[Linux]](http://virtual-home.org/release/simulator/v2.0/v2.2.4/linux_exec.zip)
   [[MacOS]](http://virtual-home.org/release/simulator/v2.0/v2.2.4/macos_exec.zip)
   [[Windows]](http://virtual-home.org/release/simulator/v2.0/v2.2.4/windows_exec.zip)
2. `pip install -r requirements.txt`

## If your machine has no display

1. Launch the X-server (sudo permission needed for one-time setup)

> [!TIP]
> If you have trouble launching the X-server (e.g., without sudo permission), try either of the following workarounds:
> - **Remote Desktop**: If remote desktop is available for your machine (e.g., *Windows App* [[MacOS]](https://apps.apple.com/us/app/windows-app/id1295203466) [[Windows]](https://apps.microsoft.com/detail/9n1f85v9t8bn), formerly *Windows Remote Desktop*), you can run `ps aux | grep Xorg` to find the display number you are using without launching the X-server. You may see something like `/usr/lib/xorg/Xorg :10 ...`, then the display number is `10`.
> - **Local Machine**: Running VirtualHome and MCTS planner uses very lite resources. You can try to run them on your local laptop.

2. Specify `--display` in [main.py](/main.py) accordingly

You can find more details [here](https://github.com/xavierpuigf/virtualhome/blob/master/README.md#test-simulator).
