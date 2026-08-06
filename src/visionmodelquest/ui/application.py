from __future__ import annotations

# ruff: noqa: E402, I001

import json
import os
import sys
from pathlib import Path
from typing import Any

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gio, GLib, Gtk

from visionmodelquest.config import ROOT, load_models, load_workload
from visionmodelquest.contracts import SYSTEM_SAFETY
from visionmodelquest.explorer.images import (
    ImageRecord,
    SessionImages,
    clean_stale_sessions,
)
from visionmodelquest.explorer.paths import ExplorerPaths
from visionmodelquest.explorer.prompts import (
    canonical_diff,
    compile_prompt,
    contract_instruction,
    is_canonical,
)
from visionmodelquest.explorer.reports import list_reports, load_report, safe_report_details
from visionmodelquest.explorer.repository import ExperimentRepository
from visionmodelquest.ui.controller import WorkerController
from visionmodelquest.ui.inspector import TokenInspector

APP_ID = "io.github.ozyjay.VisionModelQuest"


def _label(text: str, *, css: str | None = None, wrap: bool = True) -> Gtk.Label:
    widget = Gtk.Label(label=text, xalign=0)
    widget.set_wrap(wrap)
    widget.set_selectable(False)
    if css:
        widget.add_css_class(css)
    return widget


def _scrolled(child: Gtk.Widget, *, minimum_width: int = 260) -> Gtk.ScrolledWindow:
    scroll = Gtk.ScrolledWindow()
    scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scroll.set_min_content_width(minimum_width)
    scroll.set_child(child)
    return scroll


class ExplorerPage(Gtk.Box):
    def __init__(self, application: VisionExplorerApplication) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.application = application
        self.models = load_models()
        self.fixtures, self.questions = load_workload()
        self.image: ImageRecord | None = None
        self.inspection: dict[str, Any] | None = None
        self.last_generation: dict[str, Any] | None = None
        self._build()

    def _build(self) -> None:
        self.warning = Adw.Banner(
            title="This experiment modifies the canonical system instruction."
        )
        self.warning.set_revealed(False)
        self.warning.set_button_label("View Changes")
        self.warning.connect("button-clicked", lambda *_: self._show_prompt_diff())
        self.append(self.warning)

        left = self._configuration()
        centre = self._inspector()
        right = self._prompt_result()
        inner = Adw.OverlaySplitView()
        inner.set_content(centre)
        inner.set_sidebar(right)
        inner.set_sidebar_position(Gtk.PackType.END)
        inner.set_min_sidebar_width(320)
        inner.set_max_sidebar_width(480)
        inner.set_collapsed(False)
        outer = Adw.OverlaySplitView()
        outer.set_sidebar(left)
        outer.set_content(inner)
        outer.set_min_sidebar_width(280)
        outer.set_max_sidebar_width(390)
        outer.set_collapsed(False)
        self.append(outer)

    def _configuration(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        box.set_margin_top(18)
        box.set_margin_bottom(18)
        box.set_margin_start(14)
        box.set_margin_end(14)
        box.append(_label("Configuration", css="title-2"))

        lifecycle = Adw.PreferencesGroup(title="Model and worker")
        self.model_row = Adw.ComboRow(title="Model")
        model_names = [definition.display_name for definition in self.models.values()]
        self.model_keys = list(self.models)
        self.model_row.set_model(Gtk.StringList.new(model_names))
        default_key = (
            "qwen35-0.8b" if "qwen35-0.8b" in self.model_keys else self.model_keys[0]
        )
        self.model_row.set_selected(self.model_keys.index(default_key))
        self.model_row.connect("notify::selected", self._model_changed)
        lifecycle.add(self.model_row)
        self.identity_row = Adw.ActionRow(title="Processor and adapter")
        lifecycle.add(self.identity_row)
        self.state_row = Adw.ActionRow(title="Lifecycle", subtitle="Stopped")
        lifecycle.add(self.state_row)
        self.gpu_row = Adw.ActionRow(title="Compute", subtitle="Worker not started")
        lifecycle.add(self.gpu_row)
        offline = Adw.ActionRow(title="Offline mode", subtitle="Required and enforced")
        offline.add_suffix(Gtk.Image.new_from_icon_name("network-offline-symbolic"))
        lifecycle.add(offline)
        actions = Gtk.Box(spacing=6)
        self.load_button = Gtk.Button(label="Load Model")
        self.load_button.add_css_class("suggested-action")
        self.load_button.connect("clicked", lambda *_: self._load_model())
        self.unload_button = Gtk.Button(label="Unload")
        self.unload_button.connect("clicked", lambda *_: self._request("unload_model"))
        self.stop_button = Gtk.Button(label="Stop Worker")
        self.stop_button.connect("clicked", lambda *_: self.application.controller.stop())
        for button in (self.load_button, self.unload_button, self.stop_button):
            actions.append(button)
        lifecycle.add(actions)
        box.append(lifecycle)

        source = Adw.PreferencesGroup(title="Image")
        self.fixture_row = Adw.ComboRow(title="Curated fixture")
        self.fixture_row.set_model(
            Gtk.StringList.new([fixture.id.replace("_", " ").title() for fixture in self.fixtures.fixtures])
        )
        self.fixture_row.connect("notify::selected", self._fixture_changed)
        source.add(self.fixture_row)
        open_row = Adw.ActionRow(
            title="Local image",
            subtitle="PNG, JPEG or WebP · up to 16 MiB and 8192 × 8192",
        )
        open_button = Gtk.Button(label="Choose…")
        open_button.set_valign(Gtk.Align.CENTER)
        open_button.connect("clicked", lambda *_: self.application.open_image())
        open_row.add_suffix(open_button)
        source.add(open_row)
        self.image_row = Adw.ActionRow(title="Selected image", subtitle="None")
        source.add(self.image_row)
        box.append(source)

        generation = Adw.PreferencesGroup(title="Generation")
        self.contract_row = Adw.ComboRow(title="Response contract")
        self.contracts = ["scene_json_v1", "free_text_v1"]
        self.contract_row.set_model(Gtk.StringList.new(self.contracts))
        self.contract_row.connect("notify::selected", self._contract_changed)
        generation.add(self.contract_row)
        self.budget_row = Adw.SpinRow.new_with_range(140, 140, 1)
        self.budget_row.set_title("Visual-token budget")
        generation.add(self.budget_row)
        self.completion_row = Adw.SpinRow.new_with_range(32, 1024, 32)
        self.completion_row.set_title("Completion-token limit")
        self.completion_row.set_value(
            self.application.settings.get_uint("default-completion-token-limit")
        )
        generation.add(self.completion_row)
        self.inspect_button = Gtk.Button(label="Inspect Image")
        self.inspect_button.connect("clicked", lambda *_: self._inspect())
        generation.add(self.inspect_button)
        box.append(generation)
        return _scrolled(box, minimum_width=280)

    def _inspector(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        toolbar = Gtk.Box(spacing=6)
        toolbar.append(_label("Visual-token inspector", css="title-2"))
        toolbar.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        reset = Gtk.Button(icon_name="zoom-original-symbolic", tooltip_text="Reset view (Ctrl+0)")
        reset.connect("clicked", lambda *_: self.token_inspector.reset_view())
        toolbar.append(reset)
        self.raw_toggle = Gtk.ToggleButton(label="Raw patches")
        self.raw_toggle.connect("toggled", self._layers_changed)
        toolbar.append(self.raw_toggle)
        self.merged_toggle = Gtk.ToggleButton(label="Merged tokens", active=True)
        self.merged_toggle.connect("toggled", self._layers_changed)
        toolbar.append(self.merged_toggle)
        self.labels_toggle = Gtk.ToggleButton(label="Labels")
        self.labels_toggle.set_active(
            self.application.settings.get_boolean("inspector-labels-visible")
        )
        self.labels_toggle.connect("toggled", self._layers_changed)
        toolbar.append(self.labels_toggle)
        box.append(toolbar)
        self.geometry_label = _label("No preprocessing inspection is available yet.")
        box.append(self.geometry_label)
        self.original_picture = Gtk.Picture()
        self.original_picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.original_picture.set_size_request(-1, 120)
        self.original_picture.set_alternative_text("Original selected image")
        box.append(self.original_picture)
        self.token_inspector = TokenInspector()
        self.token_inspector.overlay_opacity = self.application.settings.get_double(
            "overlay-opacity"
        )
        self.token_inspector.set_hexpand(True)
        self.token_inspector.set_vexpand(True)
        self.token_inspector.connect("token-selected", self._token_selected)
        box.append(self.token_inspector)
        self.token_detail = _label("Select a merged region for token details.")
        self.token_detail.set_selectable(True)
        box.append(self.token_detail)
        note = _label(
            "Visual-token regions represent spatial inputs to the model. They are not "
            "semantic miniature images and do not show what the model has understood.",
            css="dim-label",
        )
        box.append(note)
        return box

    def _prompt_result(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(18)
        box.set_margin_bottom(18)
        box.set_margin_start(14)
        box.set_margin_end(14)
        box.append(_label("Prompt and result", css="title-2"))
        box.append(_label("System instruction"))
        self.system_text = Gtk.TextView()
        self.system_text.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.system_text.set_size_request(-1, 125)
        self.system_text.get_buffer().set_text(SYSTEM_SAFETY)
        self.system_text.get_buffer().connect("changed", self._prompt_changed)
        box.append(self.system_text)
        warning_actions = Gtk.Box(spacing=6)
        reset = Gtk.Button(label="Reset to Canonical")
        reset.connect("clicked", lambda *_: self._reset_prompt())
        continue_button = Gtk.Button(label="Continue with Modified Prompt")
        continue_button.connect("clicked", lambda *_: self.warning.set_revealed(False))
        warning_actions.append(reset)
        warning_actions.append(continue_button)
        box.append(warning_actions)
        box.append(_label("User question"))
        self.question_text = Gtk.TextView()
        self.question_text.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.question_text.set_size_request(-1, 90)
        default_question = self.questions.questions[0]
        self.question_text.get_buffer().set_text(default_question.text)
        box.append(self.question_text)
        box.append(_label("Response-contract instructions", css="heading"))
        self.contract_text = _label(contract_instruction("scene_json_v1"), css="dim-label")
        self.contract_text.set_selectable(True)
        box.append(self.contract_text)
        preview = Gtk.Expander(label="Compiled-message preview")
        self.preview_text = _label("")
        self.preview_text.set_selectable(True)
        preview.set_child(self.preview_text)
        box.append(preview)
        generate_row = Gtk.Box(spacing=8)
        self.generate_button = Gtk.Button(label="Generate")
        self.generate_button.add_css_class("suggested-action")
        self.generate_button.connect("clicked", lambda *_: self.generate())
        self.cancel_button = Gtk.Button(label="Cancel")
        self.cancel_button.add_css_class("destructive-action")
        self.cancel_button.connect("clicked", lambda *_: self.application.controller.cancel())
        generate_row.append(self.generate_button)
        generate_row.append(self.cancel_button)
        box.append(generate_row)
        box.append(_label("Generated output", css="heading"))
        self.output_text = Gtk.TextView(editable=False, cursor_visible=False)
        self.output_text.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.output_text.set_size_request(-1, 180)
        box.append(self.output_text)
        self.validation_label = _label("No generation result.")
        box.append(self.validation_label)
        self.timing_label = _label("Token counts and timings will appear here.", css="dim-label")
        self.timing_label.set_selectable(True)
        box.append(self.timing_label)
        self._update_preview()
        return _scrolled(box, minimum_width=330)

    def select_local_image(self, path: Path) -> None:
        try:
            self.image = self.application.session_images.import_image(path)
        except ValueError as error:
            self.application.show_error("Image rejected", str(error))
            return
        self.image_row.set_subtitle(
            f"{self.image.original_name} · {self.image.width} × {self.image.height}"
        )
        self.original_picture.set_filename(self.image.session_path)
        self.inspection = None
        self._inspect()

    def generate(self) -> None:
        if not self.image or not self.inspection:
            self.application.show_error("Image not ready", "Choose and inspect an image first.")
            return
        if self.application.controller.state.value != "model_ready":
            self.application.show_error("Model not ready", "Load the selected model first.")
            return
        self.output_text.get_buffer().set_text("")
        self.last_generation = None
        try:
            self.application.controller.request(
                "generate",
                {
                    "image_id": self.image.image_id,
                    "contract": self._contract(),
                    "system_instruction": self._buffer_text(self.system_text),
                    "user_question": self._buffer_text(self.question_text),
                    "completion_token_limit": int(self.completion_row.get_value()),
                },
            )
        except (RuntimeError, ValueError) as error:
            self.application.show_error("Generation could not start", str(error))

    def append_output(self, text: str) -> None:
        buffer = self.output_text.get_buffer()
        buffer.insert(buffer.get_end_iter(), text)

    def handle_response(self, payload: dict[str, Any]) -> None:
        operation = payload.get("operation")
        if payload.get("status") != "ok":
            error = payload.get("error", {})
            self.application.show_error("Worker operation failed", error.get("message", "Unknown error"))
            return
        result = payload.get("result", {})
        if operation == "initialise_processor":
            self.identity_row.set_subtitle(
                f"{result.get('processor_class')} · {result.get('adapter')} "
                f"{result.get('adapter_version')}"
            )
            if self.image:
                self._inspect()
            else:
                self._select_initial_fixture()
        elif operation == "inspect_image":
            self.inspection = result
            path = self.application.session_images.processed / f"{result['processed_image_id']}.png"
            self.token_inspector.set_inspection(str(path), result)
            self.geometry_label.set_text(
                f"Processed {result['processed_height']} × {result['processed_width']} "
                f"(H × W) · "
                f"raw {result['raw_grid_rows']} × {result['raw_grid_columns']} "
                f"({result['raw_grid_rows'] * result['raw_grid_columns']}) · "
                f"merged {result['merged_grid_rows']} × {result['merged_grid_columns']} · "
                f"{result['actual_visual_tokens']} visual tokens"
            )
        elif operation == "load_model":
            runtime = result.get("runtime", {})
            self.gpu_row.set_subtitle(
                f"{runtime.get('device_name', runtime.get('device', 'Ready'))} · "
                f"loaded in {result.get('load_seconds', 0):.2f} s"
            )
            self.application.notify("Model loaded", "The selected model is ready.")
        elif operation == "unload_model" and payload.get("auto_unload"):
            self.application.notify(
                "Model automatically unloaded",
                "The model was released after the configured idle period.",
            )
        elif operation == "generate":
            self.last_generation = result
            self.validation_label.set_text(
                f"Validation: {result.get('validation_state')} — "
                f"{result.get('validation_message')}"
            )
            self.timing_label.set_text(
                f"Input {result.get('input_token_count', 'n/a')} · "
                f"visual {result.get('visual_token_count', 'n/a')} · "
                f"completion {result.get('completion_token_count', 'n/a')} tokens\n"
                f"First token {self._seconds(result.get('time_to_first_token_seconds'))} · "
                f"generation {self._seconds(result.get('generation_seconds'))} · "
                f"{result.get('tokens_per_second', 0):.2f} tokens/s"
            )
            if not self.application.window.is_active():
                self.application.notify("Generation completed", "The local output is ready.")

    def update_state(self, state: str) -> None:
        self.state_row.set_subtitle(state.replace("_", " ").title())
        self.load_button.set_sensitive(state == "processor_ready")
        self.unload_button.set_sensitive(state == "model_ready")
        self.generate_button.set_sensitive(
            state == "model_ready" and self.image is not None and self.inspection is not None
        )
        self.cancel_button.set_sensitive(state == "generating")

    def experiment_payload(self, name: str, notes: str) -> dict[str, Any]:
        if not self.image:
            raise ValueError("choose an image before saving an experiment")
        definition = self.models[self._model_key()]
        return {
            "experiment_name": name,
            "model_key": definition.key,
            "model_id": definition.model_id,
            "model_revision": definition.revision,
            "adapter_name": definition.adapter,
            "adapter_version": definition.adapter_version,
            "response_contract": self._contract(),
            "system_instruction": self._buffer_text(self.system_text),
            "user_question": self._buffer_text(self.question_text),
            "visual_token_budget": int(self.budget_row.get_value()),
            "completion_token_limit": int(self.completion_row.get_value()),
            "image_reference": {
                "kind": "session",
                "sha256": self.image.sha256,
                "provenance": self.image.provenance,
            },
            "notes": notes[:2_000],
            "output_hash": (
                self.last_generation.get("output_hash") if self.last_generation else None
            ),
            "validation_state": (
                self.last_generation.get("validation_state") if self.last_generation else None
            ),
            "timing_summary": {
                key: self.last_generation.get(key) if self.last_generation else None
                for key in (
                    "preprocessing_seconds",
                    "time_to_first_token_seconds",
                    "generation_seconds",
                    "tokens_per_second",
                )
            },
            "preprocessing_inspection": self.inspection,
        }

    def load_revision(self, revision: object) -> None:
        if revision.model_key in self.model_keys:
            self.model_row.set_selected(self.model_keys.index(revision.model_key))
        if revision.response_contract in self.contracts:
            self.contract_row.set_selected(
                self.contracts.index(revision.response_contract)
            )
        self.system_text.get_buffer().set_text(revision.system_instruction)
        self.question_text.get_buffer().set_text(revision.user_question)
        self.completion_row.set_value(revision.completion_token_limit)
        digest = revision.image_reference.get("sha256")
        if digest:
            asset = self.application.paths.assets / digest / "image.png"
            if asset.is_file():
                self.select_local_image(asset)

    def _fixture_changed(self, *_args: object) -> None:
        self._select_initial_fixture()

    def _select_initial_fixture(self) -> None:
        index = self.fixture_row.get_selected()
        if index >= len(self.fixtures.fixtures):
            return
        fixture = self.fixtures.fixtures[index]
        path = ROOT / "fixtures" / fixture.image
        if self.image and self.image.original_name == path.name:
            return
        try:
            self.image = self.application.session_images.import_image(
                path,
                fixture=True,
                provenance=f"Checked-in fixture {fixture.id}",
            )
        except (ValueError, FileExistsError) as error:
            self.application.show_error("Fixture unavailable", str(error))
            return
        self.image_row.set_subtitle(f"{fixture.id} · {self.image.width} × {self.image.height}")
        self.original_picture.set_filename(self.image.session_path)
        allowed = set(fixture.allowed_questions)
        question = next(
            (item for item in self.questions.questions if item.id in allowed),
            self.questions.questions[0],
        )
        self.question_text.get_buffer().set_text(question.text)
        if question.contract in self.contracts:
            self.contract_row.set_selected(self.contracts.index(question.contract))
        self._inspect()

    def _inspect(self) -> None:
        if not self.image:
            return
        if self.application.controller.state.value not in {"processor_ready", "model_ready"}:
            return
        self._request(
            "inspect_image",
            {
                "image_id": self.image.image_id,
                "visual_token_budget": int(self.budget_row.get_value()),
            },
        )

    def _load_model(self) -> None:
        self._request("load_model")

    def _request(self, operation: str, payload: dict[str, Any] | None = None) -> None:
        try:
            self.application.controller.request(operation, payload)
        except (RuntimeError, ValueError) as error:
            self.application.show_error("Worker unavailable", str(error))

    def _model_changed(self, *_args: object) -> None:
        key = self._model_key()
        self.inspection = None
        definition = self.models[key]
        budget = definition.visual_token_budget or 140
        self.budget_row.set_range(budget, budget)
        self.budget_row.set_value(budget)
        maximum = definition.maximum_completion_tokens
        self.completion_row.set_range(32, maximum)
        self.completion_row.set_value(min(256, maximum))
        self.application.controller.change_model(key)

    def _contract_changed(self, *_args: object) -> None:
        self.contract_text.set_text(contract_instruction(self._contract()))
        self._update_preview()

    def _prompt_changed(self, *_args: object) -> None:
        modified = not is_canonical(self._buffer_text(self.system_text))
        self.warning.set_revealed(modified)
        self._update_preview()

    def _update_preview(self) -> None:
        try:
            prompt = compile_prompt(
                self._buffer_text(self.system_text),
                self._buffer_text(self.question_text),
                self._contract(),
            )
            self.preview_text.set_text(
                f"SYSTEM\n{prompt.system_instruction}\n\nUSER\n{prompt.user_content}"
            )
        except ValueError:
            self.preview_text.set_text("Complete the prompt to preview its messages.")

    def _reset_prompt(self) -> None:
        self.system_text.get_buffer().set_text(SYSTEM_SAFETY)

    def _show_prompt_diff(self) -> None:
        self.application.show_text_dialog(
            "Changes from canonical instruction",
            canonical_diff(self._buffer_text(self.system_text)) or "No changes.",
        )

    def _layers_changed(self, *_args: object) -> None:
        self.token_inspector.show_raw = self.raw_toggle.get_active()
        self.token_inspector.show_merged = self.merged_toggle.get_active()
        self.token_inspector.show_labels = self.labels_toggle.get_active()
        self.token_inspector.queue_draw()

    def _token_selected(self, _inspector: TokenInspector, token: object) -> None:
        if token is None:
            self.token_detail.set_text("Select a merged region for token details.")
            return
        patches = ", ".join(f"({row}, {column})" for row, column in token.raw_patches)
        processed = token.processed_bounds
        source = token.source_bounds
        self.token_detail.set_text(
            f"Token {token.sequence_index} · row {token.row}, column {token.column}\n"
            f"Processed bounds: x {processed.x:.1f}, y {processed.y:.1f}, "
            f"{processed.width:.1f} × {processed.height:.1f}\n"
            f"Source bounds: x {source.x:.2f}, y {source.y:.2f}, "
            f"{source.width:.2f} × {source.height:.2f}\n"
            f"Raw patches: {patches}"
        )

    def _model_key(self) -> str:
        return self.model_keys[self.model_row.get_selected()]

    def _contract(self) -> str:
        return self.contracts[self.contract_row.get_selected()]

    @staticmethod
    def _buffer_text(view: Gtk.TextView) -> str:
        buffer = view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)

    @staticmethod
    def _seconds(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.3f} s"


class ExperimentsPage(Gtk.Box):
    def __init__(self, application: VisionExplorerApplication) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.application = application
        self.selected_experiment_id: str | None = None
        self.set_margin_top(18)
        self.set_margin_bottom(18)
        self.set_margin_start(18)
        self.set_margin_end(18)
        header = Gtk.Box(spacing=8)
        header.append(_label("Experiments", css="title-1"))
        self.search = Gtk.SearchEntry(placeholder_text="Search experiments")
        self.search.set_hexpand(True)
        self.search.connect("search-changed", lambda *_: self.refresh())
        header.append(self.search)
        save = Gtk.Button(label="Save New Revision")
        save.add_css_class("suggested-action")
        save.connect("clicked", lambda *_: self.save_revision())
        edit = Gtk.Button(label="Edit as New Revision")
        edit.connect("clicked", lambda *_: self.edit_selected(duplicate=False))
        duplicate = Gtk.Button(label="Duplicate")
        duplicate.connect("clicked", lambda *_: self.edit_selected(duplicate=True))
        header.append(duplicate)
        header.append(edit)
        header.append(save)
        self.append(header)
        split = Adw.NavigationSplitView()
        self.list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.list.connect("row-selected", self._selected)
        split.set_sidebar(
            Adw.NavigationPage.new(_scrolled(self.list), "Experiment revisions")
        )
        self.detail = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        self.detail.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        split.set_content(
            Adw.NavigationPage.new(
                _scrolled(self.detail, minimum_width=480),
                "Revision details",
            )
        )
        split.set_min_sidebar_width(280)
        split.set_max_sidebar_width(420)
        split.set_vexpand(True)
        self.append(split)
        self.refresh()

    def refresh(self) -> None:
        while child := self.list.get_first_child():
            self.list.remove(child)
        query = self.search.get_text().casefold() if hasattr(self, "search") else ""
        for revision in self.application.experiments.list_revisions():
            if query and query not in revision.experiment_name.casefold():
                continue
            row = Gtk.ListBoxRow()
            row.revision = revision
            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
            content.set_margin_top(8)
            content.set_margin_bottom(8)
            content.set_margin_start(8)
            content.set_margin_end(8)
            content.append(_label(revision.experiment_name, css="heading"))
            content.append(
                _label(
                    f"Revision {revision.revision_number} · {revision.model_key}",
                    css="dim-label",
                )
            )
            row.set_child(content)
            self.list.append(row)

    def save_revision(self) -> None:
        dialog = Adw.AlertDialog(
            heading="Save immutable experiment revision",
            body="Saving creates a new revision. Existing revisions are never changed.",
        )
        name = Gtk.Entry(placeholder_text="Experiment name", text="Vision experiment")
        dialog.set_extra_child(name)
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("save", "Save")
        dialog.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)

        def completed(_dialog: Adw.AlertDialog, response: str) -> None:
            if response != "save":
                return
            try:
                payload = self.application.explorer.experiment_payload(name.get_text(), "")
                payload["image_reference"] = {
                    "kind": "asset",
                    "sha256": self.application.explorer.image.sha256,
                    "provenance": self.application.explorer.image.provenance,
                }
                revision = self.application.experiments.save(
                    payload,
                    experiment_id=self.selected_experiment_id,
                )
                self.application.session_images.save_asset(
                    self.application.explorer.image.image_id,
                    self.application.paths.assets,
                )
                self.application.toast(
                    f"Saved {revision.experiment_name}, revision {revision.revision_number}"
                )
                self.refresh()
            except (OSError, ValueError) as error:
                self.application.show_error("Experiment was not saved", str(error))

        dialog.connect("response", completed)
        dialog.present(self.application.window)

    def edit_selected(self, *, duplicate: bool) -> None:
        row = self.list.get_selected_row()
        if row is None:
            self.application.toast("Select an experiment revision first.")
            return
        self.application.explorer.load_revision(row.revision)
        self.selected_experiment_id = (
            None if duplicate else row.revision.experiment_id
        )
        self.application.stack.set_visible_child_name("explorer")
        self.application.toast(
            "Loaded as a duplicate."
            if duplicate
            else "Edits will be saved as a new immutable revision."
        )

    def _selected(self, _list: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if row is None:
            return
        payload = row.revision.as_dict()
        self.selected_experiment_id = row.revision.experiment_id
        self.detail.get_buffer().set_text(json.dumps(payload, indent=2, sort_keys=True))


class ReportsPage(Gtk.Box):
    def __init__(self, application: VisionExplorerApplication) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.application = application
        self.selected_path: Path | None = None
        self.revealed: set[Path] = set()
        self.set_margin_top(18)
        self.set_margin_bottom(18)
        self.set_margin_start(18)
        self.set_margin_end(18)
        header = Gtk.Box(spacing=8)
        header.append(_label("Benchmark reports", css="title-1"))
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", tooltip_text="Refresh reports")
        refresh.connect("clicked", lambda *_: self.refresh())
        header.append(refresh)
        self.reveal = Gtk.Button(label="Reveal Local Captured Content")
        self.reveal.connect("clicked", lambda *_: self._confirm_reveal())
        header.append(self.reveal)
        self.append(header)
        self.banner = Adw.Banner(
            title="Reports are read-only benchmark evidence. Captured model output is hidden."
        )
        self.banner.set_revealed(True)
        self.append(self.banner)
        split = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.list = Gtk.ListBox()
        self.list.connect("row-selected", self._selected)
        split.set_start_child(_scrolled(self.list))
        self.detail = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        self.detail.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        split.set_end_child(_scrolled(self.detail, minimum_width=500))
        split.set_position(340)
        split.set_vexpand(True)
        self.append(split)
        self.refresh()

    def refresh(self) -> None:
        while child := self.list.get_first_child():
            self.list.remove(child)
        for report in list_reports(ROOT / "reports"):
            row = Gtk.ListBoxRow()
            row.report_path = report.path
            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
            content.set_margin_top(8)
            content.set_margin_bottom(8)
            content.set_margin_start(8)
            content.set_margin_end(8)
            content.append(_label(f"{report.run_id} · {report.preset}", css="heading"))
            content.append(
                _label(
                    f"{report.status.title()} · {report.model_count} model(s)"
                    + (" · local capture" if report.quality_capture else ""),
                    css="dim-label",
                )
            )
            row.set_child(content)
            self.list.append(row)

    def _selected(self, _list: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if row is None:
            return
        self.selected_path = row.report_path
        self._render()

    def _render(self) -> None:
        if not self.selected_path:
            return
        try:
            report = load_report(self.selected_path)
            text = safe_report_details(
                report,
                reveal_capture=self.selected_path in self.revealed,
            )
            self.detail.get_buffer().set_text(text)
        except ValueError as error:
            self.detail.get_buffer().set_text(str(error))

    def _confirm_reveal(self) -> None:
        if not self.selected_path:
            return
        dialog = Adw.AlertDialog(
            heading="Reveal local captured content?",
            body=(
                "This report may contain locally captured model output. It will be shown "
                "only for this application session."
            ),
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("reveal", "Reveal")
        dialog.set_response_appearance("reveal", Adw.ResponseAppearance.DESTRUCTIVE)

        def completed(_dialog: Adw.AlertDialog, response: str) -> None:
            if response == "reveal" and self.selected_path:
                self.revealed.add(self.selected_path)
                self._render()

        dialog.connect("response", completed)
        dialog.present(self.application.window)


class VisionExplorerApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.paths = ExplorerPaths.from_environment()
        self.paths.prepare()
        self.settings = Gio.Settings.new(APP_ID)
        clean_stale_sessions(self.paths.sessions)
        self.session_images = SessionImages(self.paths.sessions)
        self.experiments = ExperimentRepository(self.paths.experiments)
        self.window: Adw.ApplicationWindow
        self.explorer: ExplorerPage
        self.controller: WorkerController
        self.toast_overlay: Adw.ToastOverlay
        self._error_dialog: Adw.AlertDialog | None = None
        self._worker_error_dialog: Adw.AlertDialog | None = None
        self.connect("shutdown", self._shutdown)
        self._install_actions()

    def do_activate(self) -> None:
        if hasattr(self, "window"):
            self.window.present()
            return
        self.window = Adw.ApplicationWindow(application=self)
        self.window.set_title("Vision Processing Explorer")
        self.window.set_default_size(1500, 900)
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        self.stack = Adw.ViewStack()
        switcher = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        header.set_title_widget(switcher)
        menu = Gio.Menu()
        menu.append("Preferences", "app.preferences")
        menu.append("About", "app.about")
        menu.append("Quit", "app.quit")
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        header.pack_end(menu_button)
        toolbar.add_top_bar(header)
        bottom_switcher = Adw.ViewSwitcherBar(stack=self.stack, reveal=True)
        toolbar.add_bottom_bar(bottom_switcher)

        self.explorer = ExplorerPage(self)
        self.experiments_page = ExperimentsPage(self)
        self.reports_page = ReportsPage(self)
        self.stack.add_titled_with_icon(
            self.explorer, "explorer", "Explorer", "image-x-generic-symbolic"
        )
        self.stack.add_titled_with_icon(
            self.experiments_page, "experiments", "Experiments", "document-save-symbolic"
        )
        self.stack.add_titled_with_icon(
            self.reports_page, "reports", "Reports", "view-list-symbolic"
        )
        toolbar.set_content(self.stack)
        self.toast_overlay = Adw.ToastOverlay(child=toolbar)
        self.window.set_content(self.toast_overlay)
        application_pid = self.paths.runtime / "application.pid"
        application_pid.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "application_id": APP_ID,
                    "session_id": self.session_images.session_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        application_pid.chmod(0o600)
        worker_python = ROOT / ".venv-rocm72" / "bin" / "python"
        if not worker_python.exists():
            worker_python = ROOT / ".venv" / "bin" / "python"
        self.controller = WorkerController(
            python=worker_python,
            model_key=self.explorer._model_key(),
            session_root=self.session_images.root,
            runtime_root=self.paths.runtime,
            log_root=self.paths.logs,
            idle_seconds=self.settings.get_uint("idle-unload-minutes") * 60,
        )
        self.controller.connect("state-changed", self._state_changed)
        self.controller.connect("response", self._response)
        self.controller.connect(
            "output-fragment", lambda _controller, text: self.explorer.append_output(text)
        )
        self.controller.connect(
            "worker-error", lambda _controller, message: self._worker_error(message)
        )
        self.explorer.update_state(self.controller.state.value)
        self.window.present()
        self.controller.start()

    def open_image(self) -> None:
        dialog = Gtk.FileDialog(title="Choose an image")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        image_filter = Gtk.FileFilter(name="PNG, JPEG and WebP images")
        for mime in ("image/png", "image/jpeg", "image/webp"):
            image_filter.add_mime_type(mime)
        filters.append(image_filter)
        dialog.set_filters(filters)

        def selected(source: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
            try:
                selected_file = source.open_finish(result)
            except GLib.Error as error:
                if not error.matches(Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED):
                    self.show_error("Image could not be opened", error.message)
                return
            path = selected_file.get_path()
            if path:
                self.explorer.select_local_image(Path(path))

        dialog.open(self.window, None, selected)

    def notify(self, title: str, body: str) -> None:
        notification = Gio.Notification.new(title)
        notification.set_body(body)
        self.send_notification(None, notification)

    def toast(self, text: str) -> None:
        self.toast_overlay.add_toast(Adw.Toast(title=text, timeout=4))

    def show_error(self, heading: str, body: str) -> None:
        if self._error_dialog is not None:
            self._error_dialog.set_heading(heading)
            self._error_dialog.set_body(body)
            return
        dialog = Adw.AlertDialog(heading=heading, body=body)
        dialog.add_response("close", "Close")

        def closed(*_args: object) -> None:
            self._error_dialog = None

        dialog.connect("response", closed)
        self._error_dialog = dialog
        dialog.present(self.window)

    def show_text_dialog(self, heading: str, text: str) -> None:
        dialog = Adw.AlertDialog(heading=heading)
        view = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        view.get_buffer().set_text(text)
        scroll = _scrolled(view, minimum_width=600)
        scroll.set_min_content_height(300)
        dialog.set_extra_child(scroll)
        dialog.add_response("close", "Close")
        dialog.present(self.window)

    def _install_actions(self) -> None:
        actions = {
            "new-experiment": lambda *_: self._new_experiment(),
            "open-image": lambda *_: self.open_image(),
            "load-model": lambda *_: self.explorer._load_model(),
            "unload-model": lambda *_: self.explorer._request("unload_model"),
            "stop-worker": lambda *_: self.controller.stop(),
            "save-revision": lambda *_: self.experiments_page.save_revision(),
            "generate": lambda *_: self.explorer.generate(),
            "cancel": lambda *_: self.controller.cancel(),
            "reset-zoom": lambda *_: self.explorer.token_inspector.reset_view(),
            "preferences": lambda *_: self._preferences(),
            "about": lambda *_: self._about(),
            "quit": lambda *_: self.quit(),
        }
        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            self.add_action(action)
        for action, accelerators in {
            "app.new-experiment": ["<Control>n"],
            "app.open-image": ["<Control>o"],
            "app.save-revision": ["<Control>s"],
            "app.generate": ["<Control>Return"],
            "app.cancel": ["Escape"],
            "app.reset-zoom": ["<Control>0"],
            "app.quit": ["<Control>q"],
        }.items():
            self.set_accels_for_action(action, accelerators)

    def _state_changed(self, _controller: WorkerController, state: str) -> None:
        self.explorer.update_state(state)

    def _response(self, _controller: WorkerController, payload: object) -> None:
        if isinstance(payload, dict):
            self.explorer.handle_response(payload)

    def _worker_error(self, message: str) -> None:
        self.notify("Vision worker failed", message)
        if self._worker_error_dialog is not None:
            self._worker_error_dialog.set_body(message)
            return
        dialog = Adw.AlertDialog(heading="Inference worker failed", body=message)
        dialog.add_response("close", "Close")

        def closed(*_args: object) -> None:
            self._worker_error_dialog = None

        dialog.connect("response", closed)
        self._worker_error_dialog = dialog
        dialog.present(self.window)

    def _new_experiment(self) -> None:
        self.stack.set_visible_child_name("explorer")
        self.experiments_page.selected_experiment_id = None
        self.explorer._reset_prompt()
        self.explorer.output_text.get_buffer().set_text("")
        self.explorer.last_generation = None

    def _preferences(self) -> None:
        dialog = Adw.PreferencesDialog(title="Preferences")
        page = Adw.PreferencesPage(title="Explorer")
        group = Adw.PreferencesGroup(title="Session defaults")
        idle = Adw.SpinRow.new_with_range(1, 60, 1)
        idle.set_title("Idle unload duration (minutes)")
        idle.set_value(max(1, self.controller.idle_seconds // 60))
        opacity = Adw.SpinRow.new_with_range(0.1, 1.0, 0.05)
        opacity.set_title("Token-overlay opacity")
        opacity.set_digits(2)
        opacity.set_value(self.explorer.token_inspector.overlay_opacity)
        group.add(idle)
        group.add(opacity)
        page.add(group)
        dialog.add(page)

        def closed(*_args: object) -> None:
            self.controller.idle_seconds = int(idle.get_value() * 60)
            self.explorer.token_inspector.overlay_opacity = opacity.get_value()
            self.settings.set_uint("idle-unload-minutes", int(idle.get_value()))
            self.settings.set_double("overlay-opacity", opacity.get_value())
            self.settings.set_boolean(
                "inspector-labels-visible",
                self.explorer.labels_toggle.get_active(),
            )
            self.explorer.token_inspector.queue_draw()

        dialog.connect("closed", closed)
        dialog.present(self.window)

    def _about(self) -> None:
        dialog = Adw.AboutDialog(
            application_name="Vision Processing Explorer",
            application_icon=APP_ID,
            developer_name="VisionModelQuest contributors",
            version="0.1.0",
            comments="Offline visual-language model experiments for Fedora GNOME.",
        )
        dialog.present(self.window)

    def _shutdown(self, *_args: object) -> None:
        if hasattr(self, "controller"):
            self.controller.close()
        self.session_images.close()
        (self.paths.runtime / "application.pid").unlink(missing_ok=True)


def main() -> None:
    application = VisionExplorerApplication()
    raise SystemExit(application.run(sys.argv))


if __name__ == "__main__":
    main()
