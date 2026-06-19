#!/usr/bin/env python3

from visualization_msgs.msg import MarkerArray, Marker
import math
import threading

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped


def make_map_image(data):

    h, w = data.shape

    img = np.ones((h, w, 3), dtype=np.float32) * 0.55

    free = (data >= 0) & (data < 50)
    occupied = data >= 50

    img[free] = 1.0

    mid = (data >= 0) & (data < 100)

    if mid.any():

        ratio = data[mid] / 100.0

        img[mid,0] = 1-ratio
        img[mid,1] = 1-ratio
        img[mid,2] = 1-ratio

    img[occupied] = 0.10

    return img


class MapViewerNode(Node):

    def __init__(self):

        super().__init__("map_viewer")

        self.latest_map = None
        self.map_info = None

        self.robot_x = None
        self.robot_y = None
        self.robot_theta = 0.0

        self.landmarks = []

        self.rrt_tree = None
        self.rrt_frontiers = None
        self.rrt_path = None

        self._lock = threading.Lock()

        self.create_subscription(
            OccupancyGrid,
            "/slam_map",
            self._map_cb,
            1
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/slam_pose",
            self._pose_cb,
            10
        )

        self.create_subscription(
            MarkerArray,
            "/slam_landmarks",
            self._landmark_cb,
            10
        )

        self.create_subscription(
            Marker,
            "/rrt_tree",
            self._rrt_tree_cb,
            1
        )

        self.create_subscription(
            Marker,
            "/rrt_frontiers",
            self._frontier_cb,
            1
        )

        self.create_subscription(
            Marker,
            "/rrt_selected_path",
            self._path_cb,
            1
        )

        self.get_logger().info(
            "MapViewer ready"
        )

    def _map_cb(self,msg):

        w = msg.info.width
        h = msg.info.height

        data=np.array(
            msg.data,
            dtype=np.int8
        ).reshape((h,w))

        with self._lock:

            self.latest_map=data
            self.map_info=msg.info


    def _pose_cb(self,msg):

        with self._lock:

            self.robot_x=msg.pose.pose.position.x
            self.robot_y=msg.pose.pose.position.y

            q=msg.pose.pose.orientation

            siny=2*(q.w*q.z + q.x*q.y)
            cosy=1-2*(q.y*q.y + q.z*q.z)

            self.robot_theta=math.atan2(
                siny,
                cosy
            )


    def _landmark_cb(self,msg):

        landmarks=[]

        for marker in msg.markers:

            if marker.action==marker.DELETEALL:
                continue

            landmarks.append(

                (
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.id
                )
            )

        with self._lock:
            self.landmarks=landmarks


    def _rrt_tree_cb(self,msg):

        with self._lock:
            self.rrt_tree=list(msg.points)


    def _frontier_cb(self,msg):

        with self._lock:
            self.rrt_frontiers=list(msg.points)


    def _path_cb(self,msg):

        with self._lock:
            self.rrt_path=list(msg.points)


    def get_data(self):

        with self._lock:

            return (
                self.latest_map,
                self.map_info,
                self.robot_x,
                self.robot_y,
                self.robot_theta,
                self.landmarks.copy(),
                self.rrt_tree,
                self.rrt_frontiers,
                self.rrt_path
            )


class MapViewer:

    def __init__(self,node):

        self.node=node

        self.traj_x=[]
        self.traj_y=[]

        self.setup()


    def setup(self):

        self.fig,self.ax=plt.subplots(
            figsize=(9,9)
        )

        placeholder=np.zeros(
            (100,100,3)
        )

        self.im=self.ax.imshow(
            placeholder,
            origin='lower'
        )

        self.robot_dot,=self.ax.plot(
            [],
            [],
            'bo'
        )

        self.traj_line,=self.ax.plot(
            [],
            [],
            color='orange'
        )

        self.arrow=None

        self.landmark_scatter=self.ax.scatter(
            [],
            [],
            c='red',
            marker='x'
        )

        self.landmark_labels=[]

        self.rrt_tree_lines=[]
        self.path_lines=[]

        self.frontier_scatter=None


    def update(self,_):

        (
            data,
            info,
            rx,
            ry,
            rt,
            landmarks,
            rrt_tree,
            rrt_frontiers,
            rrt_path

        )=self.node.get_data()


        if data is None:
            return


        img=make_map_image(data)

        ox=info.origin.position.x
        oy=info.origin.position.y

        res=info.resolution

        w=info.width
        h=info.height


        self.im.set_data(img)

        self.im.set_extent([

            ox,
            ox+w*res,
            oy,
            oy+h*res

        ])


        for line in self.rrt_tree_lines:
            line.remove()

        for line in self.path_lines:
            line.remove()

        self.rrt_tree_lines=[]
        self.path_lines=[]


        if self.frontier_scatter:

            self.frontier_scatter.remove()
            self.frontier_scatter=None


        if rrt_tree:

            for i in range(
                0,
                len(rrt_tree)-1,
                2
            ):

                p1=rrt_tree[i]
                p2=rrt_tree[i+1]

                line,=self.ax.plot(

                    [p1.x,p2.x],
                    [p1.y,p2.y],

                    color='cyan',
                    linewidth=0.5,
                    alpha=0.5
                )

                self.rrt_tree_lines.append(
                    line
                )


        if rrt_frontiers:

            xs=[p.x for p in rrt_frontiers]
            ys=[p.y for p in rrt_frontiers]

            self.frontier_scatter=(
                self.ax.scatter(
                    xs,
                    ys,
                    c='yellow',
                    s=25
                )
            )


        if rrt_path:

            xs=[p.x for p in rrt_path]
            ys=[p.y for p in rrt_path]

            line,=self.ax.plot(
                xs,
                ys,
                color='red',
                linewidth=3
            )

            self.path_lines.append(
                line
            )


        if rx is not None:

            self.robot_dot.set_data(
                [rx],
                [ry]
            )

            self.traj_x.append(rx)
            self.traj_y.append(ry)

            self.traj_line.set_data(
                self.traj_x,
                self.traj_y
            )


        self.fig.canvas.draw_idle()


    def run(self):

        from matplotlib.animation import FuncAnimation

        self.anim=FuncAnimation(
            self.fig,
            self.update,
            interval=200
        )

        plt.show()


def main():

    rclpy.init()

    node=MapViewerNode()

    thread=threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    thread.start()

    viewer=MapViewer(node)

    viewer.run()

    node.destroy_node()

    rclpy.shutdown()


if __name__=="__main__":
    main()