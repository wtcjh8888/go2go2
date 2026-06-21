#!/bin/bash
# Go2 导航系统测试脚本
#
# 用法:
#   ./test/run_tests.sh              # 运行所有测试
#   ./test/run_tests.sh utils        # 只测试工具函数
#   ./test/run_tests.sh planner      # 只测试路径规划器
#   ./test/run_tests.sh obstacle     # 只测试避障模块
#   ./test/run_tests.sh controller   # 只测试运动控制器
#   ./test/run_tests.sh -v           # 详细输出

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PKG_DIR"

# 需要 ROS2 环境
source /home/w/C206Go2/install/setup.bash 2>/dev/null || true

VERBOSE=""
MODULE=""
for arg in "$@"; do
    case "$arg" in
        -v|--verbose) VERBOSE="-v" ;;
        utils) MODULE="test_utils" ;;
        planner) MODULE="test_path_planner" ;;
        obstacle) MODULE="test_obstacle_avoider" ;;
        controller) MODULE="test_motion_controller" ;;
        *) echo "未知参数: $arg"; echo "可用: utils, planner, obstacle, controller, -v"; exit 1 ;;
    esac
done

echo "=== Go2 导航系统测试 ==="
echo ""

if [ -n "$MODULE" ]; then
    echo "运行模块: $MODULE"
    python3 -m pytest test/${MODULE}.py $VERBOSE -x --tb=short
else
    echo "运行所有测试..."
    python3 -m pytest test/ $VERBOSE -x --tb=short
fi

echo ""
echo "=== 测试完成 ==="
