#!/bin/bash
# ---------------------------------------------------------------------------
# ALFRED Environment Debug and Initialization Script (Acknowledgment Edition)
# ---------------------------------------------------------------------------
# 这是一个测试性脚本，用于在无图形界面的环境中（headless）验证和调试 AI2THOR / ALFRED。
# 本脚本仅作演示与参考，不保证适用于所有场景。使用者需根据自身需求进行调整和完善。
# 如果运行过程中出现问题，请根据提示信息、堆栈日志进行进一步排查和修改。
#
# ---------------------------------------------------------------------------

# 可选：如果环境中没有 Xvfb，则需要安装并启动。如果你已经有了其他可用的 X server，可以注释下列命令。
# echo "启动虚拟显示 Xvfb..."
# Xvfb :2 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
# XVFB_PID=$!
# sleep 2  # 给 Xvfb 一点时间来启动

# 3) 安装依赖（可根据需要启用或注释）
# echo "安装 Python 依赖..."
# pip install --upgrade pyvirtualdisplay
# pip install --upgrade ai2thor==5.0.0

echo "=================== Testing AI2THOR Controller ==================="

# 4) 测试 AI2THOR 的基本功能
python -c "
import os
import ai2thor.controller

# 可选平台: 默认是局部渲染，如果你的环境支持 cloud rendering，可以配置：
from ai2thor.platform import CloudRendering
platform = CloudRendering
# 这里为了演示，先用默认平台即可:

# 以下仅用于演示，需要的话你可以根据分辨率需求自行修改
width = 640
height = 480
fov = 90

print('DISPLAY env var:', os.environ.get('DISPLAY'))

try:
    print('尝试初始化 AI2THOR Controller...')
    controller = ai2thor.controller.Controller(
        width=width,
        height=height,
        fieldOfView=fov,
        platform=platform,
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=False,
        renderInstanceSegmentation=False
    )
    print('Controller 初始化成功！')

    print('尝试加载场景 FloorPlan1...')
    controller.reset('FloorPlan1')
    print('场景加载成功！')

    print('一切正常，停止 Controller...')
    controller.stop()
    print('AI2THOR 测试已顺利完成！')

except Exception as e:
    print('ERROR: 出现异常:', str(e))
    import traceback
    traceback.print_exc()
"

# 可选：如果使用了上面启动的 Xvfb，可以在脚本结束后杀掉它
# kill $XVFB_PID

echo "=================== Script Execution Complete ==================="
echo "【免责声明】此脚本仅用于演示与调试，不保证适应任何特定环境。请根据实际情况进行修改。"
