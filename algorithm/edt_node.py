from enum import Enum
from typing import Union, Optional


class NodeType(Enum):
    ROOT = 0
    NORMAL = 1
    LEAF = 2


class Node:
    def __init__(
            self,
            parent: 'Node',
            selected_attribute: str,
            threshold: Union[float, str, int],
            node_type: NodeType,
            left_child: Optional['Node'] = None,
            right_child: Optional['Node'] = None,
            label: Optional[str] = None
    ):
        self.parent: Node = parent
        self.attribute: str = selected_attribute
        self.threshold: Union[float, str, int] = threshold
        self.node_type: NodeType = node_type
        self.left_child: Optional[Node] = left_child
        self.right_child: Optional[Node] = right_child
        self.label: Optional[str] = label
