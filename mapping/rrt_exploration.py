#!/usr/bin/env python3

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


@dataclass
class TreeNode:
    position: Tuple[float, float]
    parent: Optional[int]


class RRTExplorer(Node):

    def __init__(
        self,
        slam,
        step_size=0.30,
        max_iterations=400,
        frontier_cluster_radius=0.5,
        occ_threshold=0.0,
        free_threshold=0.0,
        sampling_padding_cells=20,
        completion_rounds=3,
        update_period=2.0
    ):

        super().__init__("rrt_explorer")

        self.slam = slam

        self.step_size = step_size
        self.max_iterations = max_iterations
        self.frontier_cluster_radius = frontier_cluster_radius
        self.occ_threshold = occ_threshold
        self.free_threshold = free_threshold
        self.sampling_padding_cells = sampling_padding_cells
        self.completion_rounds = completion_rounds

        self.tree_nodes = []
        self.frontier_candidates = []

        self._no_frontier_rounds = 0
        self._exploration_done = False

        self.goal_pub = self.create_publisher(
            PointStamped,
            "/exploration_goal",
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            "/exploration_path",
            10
        )

        self.create_timer(update_period, self._tick)

        self.get_logger().info("RRT Explorer ready")


    def explore_step(self):

        if self._exploration_done:
            return None, None

        self._reset_tree()

        found = self._grow_tree()

        if not found:

            self._no_frontier_rounds += 1

            if self._no_frontier_rounds >= self.completion_rounds:

                self._exploration_done = True

                self.get_logger().info(
                    "Exploration complete"
                )

            return None, None

        self._no_frontier_rounds = 0

        best = self._select_best_frontier()

        path = self._backtrack_path(
            best["parent"],
            best["position"]
        )

        return best["position"], path


    def _reset_tree(self):

        x,y,_ = self.slam.pose

        self.tree_nodes = [
            TreeNode(
                position=(float(x),float(y)),
                parent=None
            )
        ]

        self.frontier_candidates=[]


    def _grow_tree(self):

        self._known_snapshot = self.slam.known_cells      
        self._grid_snapshot  = self.slam.occupancy_grid

        known_count=np.sum(self.slam.known_cells)

        self.get_logger().info(
            f"Known cells: {known_count}"
        )

        for _ in range(self.max_iterations):

            sample=self._random_sample()

            nearest_idx=self._find_nearest(sample)

            nearest=self.tree_nodes[nearest_idx].position

            new_point=self._steer(
                nearest,
                sample
            )

            c0=self._world_to_cell(*nearest)
            c1=self._world_to_cell(*new_point)

            result=self._check_edge(
                c0,
                c1
            )

            if result is None:
                continue

            if result=="free":

                self.tree_nodes.append(

                    TreeNode(
                        position=new_point,
                        parent=nearest_idx
                    )
                )

            elif result=="unknown":

                self.frontier_candidates.append({

                    "position":new_point,
                    "parent":nearest_idx
                })

        self.get_logger().info(
            f"Tree nodes: {len(self.tree_nodes)}"
        )

        self.get_logger().info(
            f"Frontiers: {len(self.frontier_candidates)}"
        )

        return len(self.frontier_candidates)>0


    def _random_sample(self):

        xmin,xmax,ymin,ymax=self._sampling_bounds()

        return (

            random.uniform(xmin,xmax),
            random.uniform(ymin,ymax)
        )


    def _sampling_bounds(self):

        known=self.slam.known_cells
        res=self.slam.map_resolution

        if not known.any():

            return (

                self.slam.map_origin_x,

                self.slam.map_origin_x+
                self.slam.map_width*res,

                self.slam.map_origin_y,

                self.slam.map_origin_y+
                self.slam.map_height*res
            )

        ys,xs=np.where(known)

        pad=self.sampling_padding_cells

        xmin=max(0,xs.min()-pad)
        xmax=min(
            self.slam.map_width,
            xs.max()+pad
        )

        ymin=max(0,ys.min()-pad)
        ymax=min(
            self.slam.map_height,
            ys.max()+pad
        )

        x0,y0=self._cell_to_world(
            xmin,
            ymin
        )

        x1,y1=self._cell_to_world(
            xmax,
            ymax
        )

        return x0,x1,y0,y1


    def _find_nearest(self,sample):

        positions=np.array([
            n.position
            for n in self.tree_nodes
        ])

        dists=np.linalg.norm(

            positions-
            np.array(sample),

            axis=1
        )

        return int(np.argmin(dists))


    def _steer(self,p0,p1):

        p0=np.array(p0)
        p1=np.array(p1)

        direction=p1-p0

        d=np.linalg.norm(direction)

        if d<1e-6:
            return tuple(p0)

        direction/=d

        step=min(
            self.step_size,
            d
        )

        p=p0+direction*step

        return (float(p[0]),float(p[1]))


    def _classify_cell(self,cx,cy):

        if not(
            0<=cx<self.slam.map_width and
            0<=cy<self.slam.map_height
        ):
            return "out"

        if not self.slam.known_cells[cy,cx]:
            return "unknown"

        value=self.slam.occupancy_grid[cy,cx]

        if value>self.occ_threshold:
            return "occupied"

        return "free"


    def _check_edge(self,c0,c1):

        cells=self._bresenham_line(
            c0[0],
            c0[1],
            c1[0],
            c1[1]
        )

        for cx,cy in cells[:-1]:

            if self._classify_cell(cx,cy)!="free":
                return None

        return self._classify_cell(
            *cells[-1]
        )


    @staticmethod
    def _bresenham_line(x0,y0,x1,y1):

        cells=[]

        dx=abs(x1-x0)
        dy=abs(y1-y0)

        sx=1 if x0<x1 else -1
        sy=1 if y0<y1 else -1

        err=dx-dy

        x=x0
        y=y0

        while True:

            cells.append((x,y))

            if x==x1 and y==y1:
                break

            e2=2*err

            if e2>-dy:
                err-=dy
                x+=sx

            if e2<dx:
                err+=dx
                y+=sy

        return cells


    def _select_best_frontier(self):

        positions=np.array([

            c["position"]

            for c in self.frontier_candidates
        ])

        best=0
        best_count=-1

        for i in range(
            len(self.frontier_candidates)
        ):

            d=np.linalg.norm(

                positions-
                positions[i],

                axis=1
            )

            count=np.sum(
                d<self.frontier_cluster_radius
            )

            if count>best_count:

                best=i
                best_count=count

        return self.frontier_candidates[best]


    def _backtrack_path(
        self,
        parent,
        leaf
    ):

        path=[leaf]

        idx=parent

        while idx is not None:

            node=self.tree_nodes[idx]

            path.append(
                node.position
            )

            idx=node.parent

        path.reverse()

        return path


    def _world_to_cell(self,x,y):

        cx=int(
            (x-self.slam.map_origin_x)/
            self.slam.map_resolution
        )

        cy=int(
            (y-self.slam.map_origin_y)/
            self.slam.map_resolution
        )

        return cx,cy


    def _cell_to_world(self,cx,cy):

        x=self.slam.map_origin_x+(
            cx*self.slam.map_resolution
        )

        y=self.slam.map_origin_y+(
            cy*self.slam.map_resolution
        )

        return x,y


    def _tick(self):

        self.get_logger().info(
            "===== RRT TICK ====="
        )

        goal,path=self.explore_step()

        if goal is None:

            self.get_logger().info(
                "No goal"
            )

            return

        gx,gy=goal

        msg=PointStamped()

        msg.header.stamp=(
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id="map"

        msg.point.x=float(gx)
        msg.point.y=float(gy)

        self.goal_pub.publish(msg)

        path_msg=Path()

        path_msg.header=msg.header

        for x,y in path:

            p=PoseStamped()

            p.header=path_msg.header

            p.pose.position=Point(
                x=float(x),
                y=float(y),
                z=0.0
            )

            path_msg.poses.append(p)

        self.path_pub.publish(path_msg)