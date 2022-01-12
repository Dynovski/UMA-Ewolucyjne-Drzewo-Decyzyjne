# Author: Zbigniew Dynowski
import numpy as np
import pandas as pd

from math import inf
from random import randint, random
from enum import Enum
from typing import Union, Optional, Dict, Any, List

import config as cfg

from data_processing.dataloader import DataLoader
from algorithm.utils import choose_node_split_params


class NodeType(Enum):
    ROOT = 0
    NORMAL = 1
    LEAF = 2


class Node:
    def __init__(
            self,
            parent: Optional['Node'],
            selected_attribute: Optional[str],
            attribute_idx: Optional[int],
            threshold: Optional[Union[float, str, int]],
            node_type: NodeType,
            left_child: Optional['Node'] = None,
            right_child: Optional['Node'] = None,
            label: Optional[str] = None
    ):
        self.parent: Optional[Node] = parent
        self.attribute: str = selected_attribute
        self.attribute_idx: int = attribute_idx
        self.threshold: Union[float, str, int] = threshold
        self.node_type: NodeType = node_type
        self.left_child: Optional[Node] = left_child
        self.right_child: Optional[Node] = right_child
        self.label: Optional[str] = label

    def copy(self) -> 'Node':
        node: 'Node' = Node(self.parent, self.attribute, self.attribute_idx,
                            self.threshold, self.node_type, label=self.label)
        if self.node_type != NodeType.LEAF:
            node.left_child = self.left_child.copy()
            node.right_child = self.right_child.copy()
            node.left_child.parent = node
            node.right_child.parent = node
        return node

    def depth_levels(self) -> int:
        if self.node_type == NodeType.LEAF:
            return 1
        return 1 + max(self.left_child.depth_levels(), self.right_child.depth_levels())

    def num_children_nodes(self) -> int:
        if self.node_type == NodeType.LEAF:
            return 0
        return self.left_child.num_children_nodes() + self.right_child.num_children_nodes()

    def predict(self, data: np.ndarray) -> str:
        node = self
        while node.node_type != NodeType.LEAF:
            attribute_value: Union[int, float, str] = data[node.attribute_idx]
            if isinstance(attribute_value, str):
                if attribute_value == node.threshold:
                    node = node.right_child
                else:
                    node = node.left_child
            else:
                if attribute_value > node.threshold:
                    node = node.right_child
                else:
                    node = node.left_child
        return node.label

    def random_node(self) -> 'Node':
        # https://stackoverflow.com/questions/32011232/randomly-select-a-node-from-a-binary-tree
        node_index = randint(0, self.num_children_nodes())
        if node_index == 0:
            return self
        elif self.left_child and node_index <= self.left_child.num_children_nodes():
            return self.left_child.random_node()
        else:
            return self.right_child.random_node()


class CandidateTree:
    def __init__(self, root: Node, fitness: float = inf):
        self.root: Node = root
        self.fitness: float = fitness

    def copy(self) -> 'CandidateTree':
        return CandidateTree(self.root.copy(), self.fitness)

    def height(self) -> int:
        return self.root.depth_levels()

    def random_subtree(self) -> Node:
        return self.root.random_node()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        labels: List[str] = [self.root.predict(x) for x in data.values]
        return np.asarray(labels)

    @staticmethod
    def generate_tree(split_prob: float, data: pd.DataFrame, labels: pd.Series) -> 'CandidateTree':
        attributes_info: Dict[str, Dict[str, Any]] = DataLoader.attributes_info(data)
        attribute, attribute_idx, threshold = choose_node_split_params(attributes_info)

        left_child_data: pd.DataFrame = data.loc[data[attribute] <= threshold]
        left_child_labels: pd.Series = labels.loc[data[attribute] <= threshold]
        right_child_data: pd.DataFrame = data.loc[data[attribute] > threshold]
        right_child_labels: pd.Series = labels.loc[data[attribute] > threshold]

        root: Node = Node(None, attribute, attribute_idx, threshold, NodeType.ROOT)
        if left_child_data.shape[0] == 0 or right_child_data.shape[0] == 0:
            # there always must be at least root and two children
            label = labels.value_counts().idxmax()
            root.left_child = Node(root, None, None, None, NodeType.LEAF, label=label)
            root.right_child = Node(root, None, None, None, NodeType.LEAF, label=label)
        else:
            root.left_child = CandidateTree.generate_subtree(root, split_prob, left_child_data, left_child_labels)
            root.right_child = CandidateTree.generate_subtree(root, split_prob, right_child_data, right_child_labels)
        return CandidateTree(root)

    @staticmethod
    def generate_subtree(parent: Node, split_prob: float,
                         data: pd.DataFrame, labels: pd.Series, depth: int = 2) -> Node:
        if random() < split_prob and depth <= cfg.MAX_TREE_DEPTH:
            attributes_info: Dict[str, Dict[str, Any]] = DataLoader.attributes_info(data)
            attribute, attribute_idx, threshold = choose_node_split_params(attributes_info)

            left_child_data: pd.DataFrame = data.loc[data[attribute] <= threshold]
            left_child_labels: pd.Series = labels.loc[data[attribute] <= threshold]
            right_child_data: pd.DataFrame = data.loc[data[attribute] > threshold]
            right_child_labels: pd.Series = labels.loc[data[attribute] > threshold]

            node: Optional[Node] = None
            if left_child_data.shape[0] == 0 or right_child_data.shape[0] == 0:
                label = labels.value_counts().idxmax()
                node: Node = Node(parent, None, None, None, NodeType.LEAF, label=label)
            else:
                node: Node = Node(parent, attribute, attribute_idx, threshold, NodeType.NORMAL)
                node.left_child = CandidateTree.generate_subtree(node, split_prob, left_child_data,
                                                                 left_child_labels, depth + 1)
                node.right_child = CandidateTree.generate_subtree(node, split_prob, right_child_data,
                                                                  right_child_labels, depth + 1)
            return node
        else:
            label = labels.value_counts().idxmax()
            return Node(parent, None, None, None, NodeType.LEAF, label=label)
