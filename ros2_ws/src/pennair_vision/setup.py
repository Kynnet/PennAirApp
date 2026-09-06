from glob import glob
from setuptools import setup

package_name = "pennair_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kyle Zheng",
    maintainer_email="kylexzheng@gmail.com",
    description="Streams a video as ROS images and publishes detected shapes.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "video_publisher = pennair_vision.video_publisher:main",
            "shape_detector = pennair_vision.shape_detector:main",
        ],
    },
)
