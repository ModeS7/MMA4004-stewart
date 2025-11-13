#!/usr/bin/env python3
"""
GUI Layout Builder for Stewart Platform Simulators

Builds GUI from declarative configuration with optional scrolling.
PyQt6 implementation.
"""

import logging
from typing import Dict, Any, Optional, List
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
                              QFrame, QSizePolicy)
from core.utils import (GUI_MAIN_LAYOUT_MARGIN, GUI_MAIN_LAYOUT_SPACING,
                        GUI_COLUMN_SPACING, GUI_SCROLLABLE_OUTER_SPACING)

logger = logging.getLogger(__name__)


class ScrollableColumn:
    """Scrollable column container using QScrollArea."""

    def __init__(self, parent: QWidget, width: Optional[int] = None, bg_color: str = '#1e1e1e') -> None:
        """
        Args:
            parent: Parent widget
            width: Fixed width in pixels (optional)
            bg_color: Background color for container
        """
        self.outer_widget = QWidget(parent)
        self.outer_layout = QVBoxLayout(self.outer_widget)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setSpacing(GUI_SCROLLABLE_OUTER_SPACING)

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
        self.inner_layout.setSpacing(GUI_COLUMN_SPACING)
        self.inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.inner_widget)
        self.outer_layout.addWidget(self.scroll_area)

    def get_container(self) -> QWidget:
        """Get the widget where modules should be added."""
        return self.inner_widget

    def get_layout(self) -> QVBoxLayout:
        """Get the layout where modules should be added."""
        return self.inner_layout

    def get_widget(self) -> QWidget:
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

    def __init__(self, parent_widget: QWidget, module_registry: Dict[str, Any]) -> None:
        """
        Args:
            parent_widget: Parent QWidget
            module_registry: Dict mapping module type names to module classes
        """
        self.parent_widget = parent_widget
        self.module_registry = module_registry
        self.modules: Dict[str, Any] = {}
        self.columns: List[Any] = []

    def build(self, layout_config: Dict[str, Any], colors: Dict[str, str],
              callbacks: Dict[str, Any]) -> Dict[str, Any]:
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
        main_layout.setContentsMargins(GUI_MAIN_LAYOUT_MARGIN, GUI_MAIN_LAYOUT_MARGIN,
                                       GUI_MAIN_LAYOUT_MARGIN, GUI_MAIN_LAYOUT_MARGIN)
        main_layout.setSpacing(GUI_MAIN_LAYOUT_SPACING)

        column_configs = layout_config.get('columns', [])

        # Create each column container (scrollable or fixed)
        for col_idx, col_config in enumerate(column_configs):
            width = col_config.get('width')
            scrollable = col_config.get('scrollable', False)

            # Scrollable column: uses QScrollArea for overflow handling
            if scrollable:
                column = ScrollableColumn(self.parent_widget, width=width,
                                          bg_color=colors.get('bg', '#1e1e1e'))
                main_layout.addWidget(column.get_widget())
                container_layout = column.get_layout()
                self.columns.append(column)
            # Fixed column: simple QWidget with VBoxLayout (no scrolling)
            else:
                column_widget = QWidget()
                column_layout = QVBoxLayout(column_widget)
                column_layout.setContentsMargins(0, 0, 0, 0)
                column_layout.setSpacing(GUI_COLUMN_SPACING)
                column_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

                if width:
                    column_widget.setFixedWidth(width)

                main_layout.addWidget(column_widget)
                container_layout = column_layout
                self.columns.append(column_widget)  # Note: stores QWidget instead of ScrollableColumn

            # Add all modules to this column's container layout
            module_configs = col_config.get('modules', [])
            for mod_config in module_configs:
                self._create_module(container_layout, mod_config, colors, callbacks)

        # Add plot panel if enabled (takes remaining horizontal space)
        if layout_config.get('plot', {}).get('enabled', False):
            self._create_plot_panel(main_layout, layout_config['plot'], colors)

        return self.modules

    def _create_module(self, parent_layout: QVBoxLayout, module_config: Dict[str, Any],
                       colors: Dict[str, str], callbacks: Dict[str, Any]) -> None:
        """
        Create a single module and add to parent layout.

        Args:
            parent_layout: Layout to add the module widget to
            module_config: Module configuration dict with 'type', 'name', and 'args'
            colors: Color scheme dict for module styling
            callbacks: Global callbacks dict for module event handlers

        Notes:
            - Logs a warning if module type is not found in registry
            - Stores created module in self.modules dict with module_name as key
        """
        module_type = module_config.get('type')
        module_name = module_config.get('name', module_type)
        module_args = module_config.get('args', {})

        if module_type not in self.module_registry:
            logger.warning(f"Module type not found: '{module_type}'. Available types: {list(self.module_registry.keys())}")
            return

        # Instantiate module from registry and create its widget
        module_class = self.module_registry[module_type]
        module = module_class(self.parent_widget, colors, callbacks, **module_args)

        # Add widget to layout if creation succeeded
        widget = module.create()
        if widget:
            parent_layout.addWidget(widget)
            self.modules[module_name] = module

    def _create_plot_panel(self, parent_layout: QHBoxLayout, plot_config: Dict[str, Any],
                           colors: Dict[str, str]) -> None:
        """
        Create plot panel container (actual plot widget created by simulator).

        Args:
            parent_layout: Main horizontal layout to add plot panel to
            plot_config: Plot configuration dict (currently unused, reserved for future)
            colors: Color scheme dict (currently unused, reserved for future)

        Notes:
            - Creates an expandable QWidget with a VBoxLayout
            - Simulator should add actual plot widget to this container's layout
            - Panel stored in self.modules['plot_panel'] for simulator access
        """
        plot_widget = QWidget()
        plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Create layout for plot widget so simulator can add plot to it
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_widget.setLayout(plot_layout)

        parent_layout.addWidget(plot_widget)
        self.modules['plot_panel'] = plot_widget

    def update_modules(self, state: Dict[str, Any]) -> None:
        """
        Update all modules with new state.

        Args:
            state: Dict containing state information for modules
        """
        for module_name, module in self.modules.items():
            if hasattr(module, 'update'):
                try:
                    module.update(state)
                except Exception as e:
                    # Log error but continue updating other modules
                    logger.debug(f"Failed to update module '{module_name}': {e}")


def create_standard_layout(scrollable_columns: bool = True, include_plot: bool = True) -> Dict[str, Any]:
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
