import numpy as np

from math import inf
from random import randint, random, choice, uniform, randrange
from enum import Enum
from typing import Union, Optional, Dict, Any, List
from sklearn.metrics import accuracy_score, matthews_corrcoef

import config as cfg


class NodeType(Enum):
    ROOT = 0
    NORMAL = 1
    LEAF = 2


class Node:
    def __init__(
            self,
            parent: Optional['Node'],
            node_type: NodeType,
            data: np.ndarray,
            labels: np.ndarray
            # selected_attribute: Optional[str],
            # attribute_idx: Optional[int],
            # threshold: Optional[Union[float, str, int]],
            # node_type: NodeType,
            # left_child: Optional['Node'] = None,
            # right_child: Optional['Node'] = None,
            # label: Optional[str] = None
    ):
        self.parent: Optional[Node] = parent
        # self.attribute: str = selected_attribute
        # self.attribute_idx = attribute_idx
        # self.threshold: Union[float, str, int] = threshold
        self.node_type: NodeType = node_type
        # self.left_child: Optional[Node] = left_child
        # self.right_child: Optional[Node] = right_child
        # self.label: Optional[str] = label
        num_attributes =



    def copy(self) -> 'Node':
        node: 'Node' = Node(self.parent, self.attribute, self.attribute_idx, self.threshold, self.node_type, label=self.label)
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
            if data[node.attribute_idx] > node.threshold:
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

    def score(self, data: np.ndarray, labels: np.ndarray) -> float:
        if self.fitness != inf:
            return self.fitness
        predictions: List[str] = [self.root.predict(x) for x in data]
        self.fitness = accuracy_score(labels, predictions)
        return self.fitness

    def predict(self, data: np.ndarray):
        labels: List[str] = []
        for i in range(data.shape[0]):
            x = data[i]
            labels.append(self.root.predict(x))
        return np.asarray(labels)

    @staticmethod
    def generate_tree(split_prob: float, data: np.ndarray, labels: np.ndarray) -> 'CandidateTree':
        num_attributes: int = data.shape[1]
        ranges = []
        for i in range(attributes):
            vals = [tmp[i] for tmp in x]
            ranges.append((min(vals), max(vals)))
        labels = np.unique(y)

        P = []
        for _ in range(self.mi):
            root = generate_subtree(self.p_split, attributes, ranges, labels)
            value = self.ga_fun(root, x, y)
            P.append(Tree(root, value))

        def _get_attributes_info(self):
            for attribute in self.attributes:
                info: Dict[str, Any] = {}
                data_types = self.train_data.dtypes
                if data_types[attribute] == 'float64' or data_types[attribute] == 'int64':
                    info['is_string'] = False
                    info['min_value'] = self.train_data[attribute].min()
                    info['max_value'] = self.train_data[attribute].max()
                else:
                    info['is_string'] = True
                    info['possible_values'] = self.train_data[attribute].unique()
                self.attribute_info_d[attribute] = info
        attributes: List[str] = list(data_info.keys())
        attribute_idx: int = randrange(len(attributes))
        attribute = attributes[attribute_idx]
        attr_data: Dict[str, Any] = data_info[attribute]
        threshold = None
        if attr_data['is_string']:
            threshold = choice(attr_data['possible_values'])
        else:
            threshold = round(uniform(attr_data['min_value'], attr_data['max_value']), 3)

        root: Node = Node(None, attribute, attribute_idx, threshold, NodeType.ROOT)
        root.left_child = CandidateTree.generate_subtree(root, split_prob, data_info, data_classes)
        root.right_child = CandidateTree.generate_subtree(root, split_prob, data_info, data_classes)
        return CandidateTree(root)

    @staticmethod
    def generate_subtree(parent: Node, split_prob: float, data_info: Dict[str, Dict[str, Any]], data_classes: List[str], depth: int = 2) -> Node:
        if random() < split_prob and depth <= cfg.MAX_TREE_DEPTH:
            attributes: List[str] = list(data_info.keys())
            attribute_idx: int = randrange(len(attributes))
            attribute = attributes[attribute_idx]
            attr_data = data_info[attribute]
            threshold = None
            if attr_data['is_string']:
                threshold = choice(attr_data['possible_values'])
            else:
                threshold = round(uniform(attr_data['min_value'], attr_data['max_value']), 3)

            node: Node = Node(parent, attribute, attribute_idx, threshold, NodeType.NORMAL)
            node.left_child = CandidateTree.generate_subtree(node, split_prob, data_info, data_classes, depth + 1)
            node.right_child = CandidateTree.generate_subtree(node, split_prob, data_info, data_classes, depth + 1)
            return node
        else:
            label = choice(data_classes)
            return Node(parent, None, None, None, NodeType.LEAF, label=label)
