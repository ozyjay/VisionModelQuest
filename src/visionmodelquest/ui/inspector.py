from __future__ import annotations

# ruff: noqa: E402, I001

import cairo
import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GObject, Gtk

from visionmodelquest.explorer.geometry import (
    Bounds,
    TokenRegion,
    ViewTransform,
    clamp_zoom,
    fit_transform,
    token_at_point,
)


class TokenInspector(Gtk.DrawingArea):
    __gsignals__ = {
        "token-selected": (GObject.SignalFlags.RUN_FIRST, None, (object,)),
    }

    def __init__(self) -> None:
        super().__init__()
        self.set_focusable(True)
        self.set_accessible_role(Gtk.AccessibleRole.GRID)
        self.update_property(
            [Gtk.AccessibleProperty.LABEL],
            ["Visual-token inspector; use arrow keys to move between token regions"],
        )
        self.set_content_width(480)
        self.set_content_height(560)
        self.surface: cairo.ImageSurface | None = None
        self.regions: tuple[TokenRegion, ...] = ()
        self.raw_rows = 0
        self.raw_columns = 0
        self.merged_rows = 0
        self.merged_columns = 0
        self.processed_width = 1
        self.processed_height = 1
        self.overlay_opacity = 0.72
        self.show_raw = False
        self.show_merged = True
        self.show_labels = False
        self.selected_index: int | None = None
        self.hover_index: int | None = None
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_origin = (0.0, 0.0)
        self.set_draw_func(self._draw)

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._motion)
        motion.connect("leave", self._leave)
        self.add_controller(motion)
        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
            | Gtk.EventControllerScrollFlags.DISCRETE
        )
        scroll.connect("scroll", self._scroll)
        self.add_controller(scroll)
        drag = Gtk.GestureDrag()
        drag.set_button(Gdk.BUTTON_PRIMARY)
        drag.connect("drag-begin", self._drag_begin)
        drag.connect("drag-update", self._drag_update)
        self.add_controller(drag)
        click = Gtk.GestureClick()
        click.connect("pressed", self._click)
        self.add_controller(click)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self._key_pressed)
        self.add_controller(keys)

    def set_inspection(self, image_path: str, inspection: dict[str, object]) -> None:
        self.surface = cairo.ImageSurface.create_from_png(image_path)
        self.processed_width = int(inspection["processed_width"])
        self.processed_height = int(inspection["processed_height"])
        self.raw_rows = int(inspection["raw_grid_rows"])
        self.raw_columns = int(inspection["raw_grid_columns"])
        self.merged_rows = int(inspection["merged_grid_rows"])
        self.merged_columns = int(inspection["merged_grid_columns"])
        self.regions = tuple(self._token(item) for item in inspection.get("tokens", []))
        self.selected_index = None
        self.reset_view()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.queue_draw()

    def select(self, index: int | None) -> None:
        if index is None:
            self.selected_index = None
            self.emit("token-selected", None)
        elif self.regions:
            self.selected_index = max(0, min(len(self.regions) - 1, index))
            self.emit("token-selected", self.regions[self.selected_index])
        self.queue_draw()

    def _transform(self, width: int, height: int) -> ViewTransform:
        fitted = fit_transform(
            width,
            height,
            self.processed_width,
            self.processed_height,
        )
        return ViewTransform(
            scale=fitted.scale * self.zoom,
            offset_x=fitted.offset_x + self.pan_x,
            offset_y=fitted.offset_y + self.pan_y,
        )

    def _draw(self, _area: Gtk.DrawingArea, context: cairo.Context, width: int, height: int) -> None:
        context.set_source_rgb(0.08, 0.08, 0.09)
        context.paint()
        if self.surface is None:
            context.set_source_rgb(0.8, 0.8, 0.8)
            context.move_to(30, height / 2)
            context.show_text("Choose an image to inspect its visual-token geometry.")
            return
        transform = self._transform(width, height)
        context.save()
        context.translate(transform.offset_x, transform.offset_y)
        context.scale(transform.scale, transform.scale)
        context.set_source_surface(self.surface)
        context.paint()
        line_width = max(0.5, 1.25 / transform.scale)
        if self.show_raw:
            self._draw_grid(
                context,
                self.raw_rows,
                self.raw_columns,
                (0.95, 0.65, 0.15, self.overlay_opacity * 0.7),
                line_width,
            )
        if self.show_merged:
            self._draw_grid(
                context,
                self.merged_rows,
                self.merged_columns,
                (0.2, 0.75, 1.0, self.overlay_opacity),
                line_width * 1.5,
            )
        for index, colour in (
            (self.hover_index, (1.0, 1.0, 1.0, 0.20)),
            (self.selected_index, (0.15, 0.8, 1.0, 0.42)),
        ):
            if index is not None and 0 <= index < len(self.regions):
                bounds = self.regions[index].processed_bounds
                context.set_source_rgba(*colour)
                context.rectangle(bounds.x, bounds.y, bounds.width, bounds.height)
                context.fill()
        if self.show_labels and self.show_merged:
            context.set_font_size(max(4, 10 / transform.scale))
            for region in self.regions:
                context.set_source_rgba(1, 1, 1, 0.9)
                context.move_to(
                    region.processed_bounds.x + 2 / transform.scale,
                    region.processed_bounds.y + 11 / transform.scale,
                )
                context.show_text(str(region.sequence_index))
        context.restore()

    def _draw_grid(
        self,
        context: cairo.Context,
        rows: int,
        columns: int,
        colour: tuple[float, float, float, float],
        width: float,
    ) -> None:
        if not rows or not columns:
            return
        context.set_source_rgba(*colour)
        context.set_line_width(width)
        for column in range(columns + 1):
            x = column * self.processed_width / columns
            context.move_to(x, 0)
            context.line_to(x, self.processed_height)
        for row in range(rows + 1):
            y = row * self.processed_height / rows
            context.move_to(0, y)
            context.line_to(self.processed_width, y)
        context.stroke()

    def _point_index(self, x: float, y: float) -> int | None:
        processed_x, processed_y = self._transform(
            self.get_width(), self.get_height()
        ).widget_to_processed(x, y)
        region = token_at_point(self.regions, processed_x, processed_y)
        return region.sequence_index if region else None

    def _motion(self, _controller: Gtk.EventControllerMotion, x: float, y: float) -> None:
        self.hover_index = self._point_index(x, y)
        self.queue_draw()

    def _leave(self, _controller: Gtk.EventControllerMotion) -> None:
        self.hover_index = None
        self.queue_draw()

    def _scroll(
        self,
        _controller: Gtk.EventControllerScroll,
        _dx: float,
        dy: float,
    ) -> bool:
        self.zoom = clamp_zoom(self.zoom * (1.15 if dy < 0 else 1 / 1.15))
        self.queue_draw()
        return True

    def _drag_begin(self, _gesture: Gtk.GestureDrag, _x: float, _y: float) -> None:
        self._drag_origin = (self.pan_x, self.pan_y)

    def _drag_update(self, _gesture: Gtk.GestureDrag, dx: float, dy: float) -> None:
        self.pan_x = self._drag_origin[0] + dx
        self.pan_y = self._drag_origin[1] + dy
        self.queue_draw()

    def _click(
        self,
        _gesture: Gtk.GestureClick,
        _count: int,
        x: float,
        y: float,
    ) -> None:
        self.grab_focus()
        self.select(self._point_index(x, y))

    def _key_pressed(
        self,
        _controller: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        _state: Gdk.ModifierType,
    ) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.select(None)
            return True
        if not self.regions:
            return False
        current = self.selected_index if self.selected_index is not None else 0
        if keyval == Gdk.KEY_Left:
            self.select(current - 1)
        elif keyval == Gdk.KEY_Right:
            self.select(current + 1)
        elif keyval == Gdk.KEY_Up:
            self.select(current - self.merged_columns)
        elif keyval == Gdk.KEY_Down:
            self.select(current + self.merged_columns)
        elif keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            self.select(current)
        else:
            return False
        return True

    @staticmethod
    def _token(payload: object) -> TokenRegion:
        if not isinstance(payload, dict):
            raise ValueError("invalid token payload")
        return TokenRegion(
            sequence_index=int(payload["sequence_index"]),
            row=int(payload["row"]),
            column=int(payload["column"]),
            processed_bounds=Bounds(**payload["processed_bounds"]),
            source_bounds=Bounds(**payload["source_bounds"]),
            raw_patches=tuple(tuple(item) for item in payload["raw_patches"]),
        )
