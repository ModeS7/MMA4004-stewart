#!/usr/bin/env python3
"""
GUI Layout Builder for Stewart Platform Simulators

Builds GUI from declarative configuration with optional scrolling.
"""

import tkinter as tk
from tkinter import ttk


class ScrollableColumn:
    """Scrollable column container using Canvas."""

    def __init__(self, parent, width=None, bg_color='#1e1e1e'):
        """
        Args:
            parent: Parent widget
            width: Fixed width in pixels (optional)
            bg_color: Background color for canvas
        """
        self.outer_frame = ttk.Frame(parent)
        if width:
            self.outer_frame.configure(width=width)
            self.outer_frame.pack_propagate(False)

        self.canvas = tk.Canvas(self.outer_frame, highlightthickness=0, bg=bg_color)
        self.scrollbar = ttk.Scrollbar(self.outer_frame, orient="vertical",
                                       command=self.canvas.yview)

        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0),
                                                       window=self.inner_frame,
                                                       anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._bind_mousewheel()

    def _on_frame_configure(self, event=None):
        """Update canvas scroll region when content size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Update inner frame width to match canvas."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _bind_mousewheel(self):
        """Bind mouse wheel for scrolling."""

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_enter(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _on_leave(event):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind("<Enter>", _on_enter)
        self.canvas.bind("<Leave>", _on_leave)

    def get_container(self):
        """Get the frame where modules should be added."""
        return self.inner_frame

    def pack(self, **kwargs):
        """Pack the outer frame."""
        self.outer_frame.pack(**kwargs)


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

    def __init__(self, root, module_registry):
        """
        Args:
            root: Root tkinter window
            module_registry: Dict mapping module type names to module classes
        """
        self.root = root
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
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        column_configs = layout_config.get('columns', [])

        for col_idx, col_config in enumerate(column_configs):
            width = col_config.get('width')
            scrollable = col_config.get('scrollable', False)

            if scrollable:
                column = ScrollableColumn(main_frame, width=width,
                                          bg_color=colors.get('bg', '#1e1e1e'))
                column.pack(side='left', fill='both',
                            expand=(width is None),
                            padx=(0 if col_idx == 0 else 5, 5))
                container = column.get_container()
                self.columns.append(column)
            else:
                column = ttk.Frame(main_frame, style='TFrame')
                if width:
                    column.configure(width=width)
                    column.pack_propagate(False)
                column.pack(side='left', fill='both',
                            expand=(width is None),
                            padx=(0 if col_idx == 0 else 5, 5))
                container = column
                self.columns.append(column)

            module_configs = col_config.get('modules', [])
            for mod_config in module_configs:
                self._create_module(container, mod_config, colors, callbacks)

        if layout_config.get('plot', {}).get('enabled', False):
            self._create_plot_panel(main_frame, layout_config['plot'], colors)

        return self.modules

    def _create_module(self, parent, module_config, colors, callbacks):
        """Create a single module and add to parent."""
        module_type = module_config.get('type')
        module_name = module_config.get('name', module_type)
        module_args = module_config.get('args', {})

        if module_type not in self.module_registry:
            print(f"Warning: Unknown module type '{module_type}'")
            return

        module_class = self.module_registry[module_type]
        module = module_class(parent, colors, callbacks, **module_args)

        frame = module.create()
        if frame:
            pack_config = module_config.get('pack', {'fill': 'x', 'pady': (0, 10)})
            frame.pack(**pack_config)
            self.modules[module_name] = module

    def _create_plot_panel(self, parent, plot_config, colors):
        """Create plot panel (actual plot created by simulator)."""
        plot_panel = ttk.Frame(parent, style='TFrame')
        plot_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        self.modules['plot_panel'] = plot_panel

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