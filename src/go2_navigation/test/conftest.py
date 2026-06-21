"""测试配置：共享 fixtures。"""

import numpy as np
import pytest


@pytest.fixture
def flat_ground_points():
    """创建一个平地场景的模拟点云。"""
    np.random.seed(42)
    points = []
    # 5m x 5m 平地，z=0.1m，每 0.05m 一个点
    for x in np.arange(0, 5, 0.05):
        for y in np.arange(0, 5, 0.05):
            noise = np.random.normal(0, 0.01, 3)
            points.append([x + noise[0], y + noise[1], 0.1 + noise[2]])
    return np.array(points, dtype=np.float32)


@pytest.fixture
def slope_points():
    """创建一个带坡道的模拟点云。"""
    np.random.seed(42)
    points = []
    # 平地区域 (0-3m)
    for x in np.arange(0, 3, 0.05):
        for y in np.arange(0, 2, 0.05):
            noise = np.random.normal(0, 0.01, 3)
            points.append([x + noise[0], y + noise[1], 0.1 + noise[2]])
    # 坡道区域 (3-5m, z 从 0.1 升到 0.5)
    for x in np.arange(3, 5, 0.05):
        z = 0.1 + (x - 3) * 0.2  # 约 11.3 度坡度
        for y in np.arange(0, 2, 0.05):
            noise = np.random.normal(0, 0.01, 3)
            points.append([x + noise[0], y + noise[1], z + noise[2]])
    return np.array(points, dtype=np.float32)


@pytest.fixture
def obstacle_points():
    """创建一个带障碍物的模拟点云。"""
    np.random.seed(42)
    points = []
    # 地面
    for x in np.arange(0, 5, 0.1):
        for y in np.arange(0, 5, 0.1):
            points.append([x, y, 0.1])
    # 障碍物（柱子，2m-3m 高）
    for x in np.arange(2, 2.5, 0.05):
        for y in np.arange(2, 2.5, 0.05):
            for z in np.arange(0.3, 2.0, 0.1):
                points.append([x, y, z])
    return np.array(points, dtype=np.float32)
