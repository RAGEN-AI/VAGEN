#!/usr/bin/env python

import subprocess
import shlex
import tempfile
import os
import sys
import time

def kill_existing_x_servers(display_num):
    """Kill any existing X server instances running on the specified display"""
    try:
        # Find processes running Xorg on the specified display
        cmd = f"ps aux | grep 'Xorg.*:{display_num}' | grep -v grep"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        
        if output:
            print(f"Found existing X server on display :{display_num}, terminating...")
            # Extract PIDs
            for line in output.split('\n'):
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    try:
                        print(f"Killing process {pid}")
                        subprocess.call(f"kill -9 {pid}", shell=True)
                    except Exception as e:
                        print(f"Error killing process {pid}: {e}")
            
            # Give it a moment to terminate
            time.sleep(1)
            print("Existing X servers terminated")
        else:
            print(f"No existing X server found on display :{display_num}")
    except subprocess.CalledProcessError:
        # No processes found
        print(f"No existing X server found on display :{display_num}")
    except Exception as e:
        print(f"Error while checking for existing X servers: {e}")

def generate_xorg_conf():
    """Generate a basic Xorg config file without NVIDIA-specific settings"""
    xorg_conf = """
Section "Device"
    Identifier     "Device0"
    Driver         "dummy"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection

Section "ServerLayout"
    Identifier     "Layout0"
    Screen 0 "Screen0" 0 0
EndSection
"""
    print("Generated Xorg configuration:")
    print(xorg_conf)
    return xorg_conf

def startx(display):
    """Start an X server on the specified display"""
    try:
        # Clean up existing X servers
        kill_existing_x_servers(display)
        
        # Create a temporary config file
        fd, path = tempfile.mkstemp()
        try:
            with open(path, "w") as f:
                f.write(generate_xorg_conf())
            
            print(f"Starting X server on display :{display}")
            command = shlex.split(f"Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config {path} :{display}")
            
            # Run Xorg
            print(f"Running command: {' '.join(command)}")
            subprocess.call(command)
        finally:
            os.close(fd)
            os.unlink(path)
            print(f"Removed temporary config file: {path}")
    except Exception as e:
        print(f"Error starting X server: {e}")
        raise

if __name__ == '__main__':
    display = 0
    if len(sys.argv) > 1:
        display = int(sys.argv[1])
    
    print(f"Starting X server on DISPLAY=:{display}")
    try:
        startx(display)
    except KeyboardInterrupt:
        print("\nX server startup interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)