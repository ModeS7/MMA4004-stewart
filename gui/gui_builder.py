#!/usr/bin/env python3
"""
GUI Layout Builder for Stewart Platform Simulators

Builds GUI from declarative configuration with optional scrolling.
PyQt6 implementation.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
                              QFrame, QSizePolicy)
from PyQt6.QtCore import Qt


class ScrollableColumn:
    """Scrollable column container using QScrollArea."""

    def __init__(self, parent, width=None, bg_color='#1e1e1e'):
        """
        Args:
            parent: Parent widget
            width: Fixed width in pixels (optional)
            bg_color: Background color for container
        """
        self.outer_widget = QWidget(parent)
        self.outer_layout = QVBoxLayout(self.outer_widget)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setSpacing(0)

        if width:
            self.outer_widget.setFixedWidth(width)

        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet(f"QScrollArea {{ background-color: {bg_color}; border: none; }}")

        # Inner container for modules
        self.inner_widget = QWidget()
        self.inner_layout = QVBoxLayout(self.inner_widget)
        self.inner_layout.setContentsMargins(0, 0, 0, 0)
        self.inner_layout.setSpacing(10)
        self.inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.inner_widget)
        self.outer_layout.addWidget(self.scroll_area)

    def get_container(self):
        """Get the widget where modules should be added."""
        return self.inner_widget

    def get_layout(self):
        """Get the layout where modules should be added."""
        return self.inner_layout

    def get_widget(self):
        """Get the outer widget for adding to parent layout."""
        return self.outer_widget


class GUIBuilder:
    """
    Build modular GUI from declarative configuration.

    Layout config format:
    {
        'columns': [
            {
                'width': 400,
                'scrollable': True,
                'modules': [
                    {'type': 'simulation_control', 'args': {...}},
                    {'type': 'ball_control', 'args': {...}},
                    ...
                ]
            },
            ...
        ],
        'plot': {'enabled': True, 'title': 'Ball Position'}
    }
    """

    def __init__(self, parent_widget, module_registry):
        """
        Args:
            parent_widget: Parent QWidget
            module_registry: Dict mapping module type names to module classes
        """
        self.parent_widget = parent_widget
        self.module_registry = module_registry
        self.modules = {}
        self.columns = []

    def build(self, layout_config, colors, callbacks):
        """
        Build GUI from layout configuration.

        Args:
            layout_config: Layout configuration dict
            colors: Color scheme dict
            callbacks: Global callbacks dict

        Returns:
            dict: References to created modules by name
        """
        main_layout = QHBoxLayout(self.parent_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        column_configs = layout_config.get('columns', [])

        for col_idx, col_config in enumerate(column_configs):
            width = col_config.get('width')
            scrollable = col_config.get('scrollable', False)

            if scrollable:
                column = ScrollableColumn(self.parent_widget, width=width,
                                          bg_color=colors.get('bg', '#1e1e1e'))
                main_layout.addWidget(column.get_widget())
                container_layout = column.get_layout()
                self.columns.append(column)
            else:
                column_widget = QWidget()
                column_layout = QVBoxLayout(column_widget)
                column_layout.setContentsMargins(0, 0, 0, 0)
                column_layout.setSpacing(10)
                column_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

                if width:
                    column_widget.setFixedWidth(width)

                main_layout.addWidget(column_widget)
                container_layout = column_layout
                self.columns.append(column_widget)

            module_configs = col_config.get('modules', [])
            for mod_config in module_configs:
                self._create_module(container_layout, mod_config, colors, callbacks)

        if layout_config.get('plot', {}).get('enabled', False):
            self._create_plot_panel(main_layout, layout_config['plot'], colors)

        return self.modules

    def _create_module(self, parent_layout, module_config, colors, callbacks):
        """Create a single module and add to parent layout."""
        module_type = module_config.get('type')
        module_name = module_config.get('name', module_type)
        module_args = module_config.get('args', {})

        if module_type not in self.module_registry:
            print(f"Module type not found: '{module_type}'")
            return

        module_class = self.module_registry[module_type]
        module = module_class(self.parent_widget, colors, callbacks, **module_args)

        widget = module.create()
        if widget:
            parent_layout.addWidget(widget)
            self.modules[module_name] = module

    def _create_plot_panel(self, parent_layout, plot_config, colors):
        """Create plot panel (actual plot created by simulator)."""
        plot_widget = QWidget()
        plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Create layout for plot widget so simulator can add plot to it
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_widget.setLayout(plot_layout)

        parent_layout.addWidget(plot_widget)
        self.modules['plot_panel'] = plot_widget

    def update_modules(self, state):
        """
        Update all modules with new state.

        Args:
            state: Dict containing state information for modules
        """
        for module in self.modules.values():
            if hasattr(module, 'update'):
                try:
                    module.update(state)
                except Exception:
                    pass


def create_standard_layout(scrollable_columns=True, include_plot=True):
    """
    Create a standard 2-column layout template.

    Args:
        scrollable_columns: Enable scrolling for columns
        include_plot: Include plot panel

    Returns:
        Layout configuration dict (to be customized by simulator)
    """
    return {
        'columns': [
            {
                'width': 400,
                'scrollable': scrollable_columns,
                'modules': []
            },
            {
                'width': 450,
                'scrollable': scrollable_columns,
                'modules': []
            }
        ],
        'plot': {
            'enabled': include_plot,
            'title': 'Ball Position (Top View)'
        }
    }
