# Adding a UI Feature (blank window + button example)

Pattern mirrors existing `show_histogram` flow (`mod.rs`, `ui_sidebar.rs`, `ui_windows.rs`).

## 1. `src/app/mod.rs` — struct field + default

Struct field (near `show_histogram: bool` around line 191):

```rust
pub(super) show_my_window: bool,
```

Default impl (near `show_histogram: false,` around line 445):

```rust
show_my_window: false,
```

## 2. `src/app/ui_sidebar.rs` — button

Inside `show_sidebar_panel` (`&mut self`, so direct mutation is fine — no `SidebarAction` needed for simple toggles):

```rust
if ui.button("My Feature").clicked() {
    self.show_my_window = true;
}
```

## 3. `src/app/ui_windows.rs` — window block

Inside `show_windows`, mirroring the histogram window block:

```rust
if self.show_my_window {
    let mut open = true;
    egui::Window::new("My Feature")
        .open(&mut open)
        .resizable(true)
        .default_size([300.0, 200.0])
        .show(ui.ctx(), |ui| {
            ui.label("Blank window content here");
        });
    if !open {
        self.show_my_window = false;
    }
}
```

`show_windows` is already called every frame, so no extra wiring is needed.
