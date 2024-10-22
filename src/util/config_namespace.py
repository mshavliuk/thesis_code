import os
from argparse import Namespace
from pathlib import Path
from types import UnionType
from typing import (
    Literal,
    get_origin,
)

import yaml
from yaml import (
    DocumentStartEvent,
    MappingNode,
    SafeLoader,
    ScalarNode,
    SequenceNode,
)


class MergeLoader(SafeLoader):
    def compose_document(self):
        event = self.peek_event()
        # Drop the DOCUMENT-START event.
        self.get_event()
        
        # Compose the root node.
        node = self.compose_node(None, None)
        
        # Drop the DOCUMENT-END event.
        self.get_event()
        
        if (isinstance(event, DocumentStartEvent)
            and event.tags is not None
            and (include_path := event.tags.get('!include!', None))):
            with open(os.path.join(os.path.dirname(self.stream.name), include_path), 'r') as f:
                loader = MergeLoader(f)
                base_node = loader.get_single_node()
                
                node = loader.merge_nodes(base_node, node)
                loader.dispose()
        
        # self.anchors = {}
        return node
    
    def merge_nodes(self, base_node: yaml.Node, new_node: yaml.Node) -> yaml.Node:
        if type(base_node) != type(new_node):
            return new_node
        elif isinstance(base_node, (ScalarNode, SequenceNode)):
            base_node.value = new_node.value
            base_node.tag = new_node.tag
        elif isinstance(base_node, MappingNode):
            base_children = {key.value: value for key, value in base_node.value}
            for new_key, new_value in new_node.value:
                if new_key.value not in base_children:
                    base_node.value.append((new_key, new_value))
                else:
                    child = base_children[new_key.value]
                    child_index = next(
                        i for i, (k, v) in enumerate(base_node.value) if k.value == new_key.value)
                    if child in set(self.anchors.values()):
                        # anchor element and name matches
                        if new_key.value in self.anchors:
                            self.merge_nodes(child, new_value)
                        else:
                            # anchor element but different name -> replace element without modifying original node
                            base_node.value[child_index] = (new_key, new_value)
                    else:
                        base_node.value[child_index] = (new_key, self.merge_nodes(child, new_value))
        
        else:
            raise ValueError(f"Cannot merge {type(base_node)} with {type(new_node)}")
        
        return base_node


class ConfigNamespace(Namespace):
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], self.__class__):
                super().__init__(**args[0].__dict__)
                return
            elif isinstance(args[0], dict):
                kwargs = args[0]
            else:
                raise ValueError(f'Invalid argument type: {type(args[0])}')
        
        super().__init__()
        annotations = {}
        for cls in self.__class__.__mro__:
            if cls == Namespace:
                break
            annotations.update(cls.__annotations__)
        for prop, field_type in annotations.items():
            if get_origin(field_type) is UnionType:
                field_types = field_type.__args__
            else:
                field_types = [field_type]
            
            if prop not in kwargs:
                if not hasattr(self, prop):
                    raise ValueError(f'Missing required field: {prop}')
                continue
            
            for _field_type in field_types:
                if isinstance(kwargs[prop], _field_type):
                    setattr(self, prop, kwargs[prop])
                    break
                elif isinstance(kwargs[prop], dict):
                    setattr(self, prop, _field_type(**kwargs[prop]))
                    break
                elif kwargs[prop] is not None:
                    setattr(self, prop, _field_type(kwargs[prop]))
                    break
            else:
                raise ValueError(f'Invalid type for field {prop}: {type(kwargs[prop])}')


class MainConfig(ConfigNamespace):
    description: str
    name: str
    stage: str
    data_config: dict
    module_config: dict
    wandb_logger: dict
    checkpoint: str | None = None
    trainer: dict
    checkpoint_callback: dict
    early_stop_callback: dict


def read_config(file_path: str) -> MainConfig:
    file = Path(file_path)
    with file.open('r') as f:
        config = yaml.load(f, Loader=MergeLoader)
    config['name'] = file.stem
    config = MainConfig(config)
    return config
