#!/usr/bin/env python3
"""
MISHA LED Controller
GUI for controlling the MISHA multi-wavelength LED illumination board over USB serial.

run with `python MISHA.py`

Protocol: send "wavelength,intensity\n"  (e.g. "470,75\n")
  - wavelength : one of the 16 supported nm values (or 0 to turn off everything)
  - intensity  : 0-100 (percent)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import threading
import time


# ── Hardware constants ────────────────────────────────────────────────────────

BAUD_RATE   = 9600
WAVELENGTHS = [365, 385, 395, 420, 450, 470, 500, 530,
               560, 590, 615, 630, 660, 730, 850, 940]


# ── Colour helpers ────────────────────────────────────────────────────────────

def _nm_to_hex(nm: int) -> str:
    """Return an approximate sRGB hex colour for a given wavelength (nm).
    UV and IR are rendered as dim violet / dark red respectively."""
    if nm < 380:
        return "#6600CC"
    elif nm < 400:
        return "#7B00DD"
    elif nm < 420:
        return "#9400D3"
    elif nm < 450:
        return "#4400FF"
    elif nm < 470:
        return "#0000FF"
    elif nm < 490:
        return "#0055FF"
    elif nm < 510:
        return "#00CFFF"
    elif nm < 540:
        return "#00FF44"
    elif nm < 560:
        return "#66FF00"
    elif nm < 580:
        return "#CCFF00"
    elif nm < 600:
        return "#FFE000"
    elif nm < 620:
        return "#FF8800"
    elif nm < 645:
        return "#FF3300"
    elif nm < 700:
        return "#FF0000"
    elif nm < 760:
        return "#990000"
    elif nm < 900:
        return "#440000"
    else:
        return "#220000"


WAVE_COLOR = {nm: _nm_to_hex(nm) for nm in WAVELENGTHS}

# Physical LED count per wavelength, derived from PCB schematic.
# 365/385/395/420 nm only have a single LED (Panel A).
# All others have A + B; 940 nm has A + B + C.
PANEL_LEDS = {
    365: "A · B",  385: "A · B",  395: "A · B",  420: "A · B",
    450: "A · B",  470: "A · B",  500: "A · B",  530: "A · B",
    560: "A · B",  590: "A · B",  615: "A · B",  630: "A · B",
    660: "A · B",  730: "A · B",  850: "A · B",  940: "A · B",
}


def _contrast_text(hex_bg: str) -> str:
    """Return '#111122' or '#FFFFFF' — whichever is more legible on hex_bg."""
    h = hex_bg.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    # Relative luminance (sRGB approximation)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#111122" if luminance > 0.45 else "#FFFFFF"


# ── Theme palette ─────────────────────────────────────────────────────────────

BG        = "#141422"   # window background
CARD      = "#1E1E30"   # panel / card background
CARD2     = "#252538"   # slightly lighter card
ACCENT    = "#E94560"   # primary accent (red-pink)
ACCENT_HV = "#FF6B82"   # accent hover
FG        = "#EAEAF4"   # primary text
FG2       = "#8888A8"   # secondary / muted text
BTN_DARK  = "#2A2A40"   # unselected button
SUCCESS   = "#00D4AA"   # green – connected
WARNING   = "#F5A623"   # amber – disconnect button
SEP       = "#2E2E44"   # separator lines

FONT_FAMILY = "Helvetica Neue"  # falls back gracefully on all platforms


# ── Custom button widget ──────────────────────────────────────────────────────

class _FlatButton(tk.Frame):
    """Label inside a Frame — fully respects custom bg colors on macOS
    (tk.Button background is overridden by the Aqua theme on macOS)."""

    def __init__(self, parent, text, command,
                 bg_def, bg_hov, fg_col, font, width,
                 padx=10, pady=7):
        super().__init__(parent, bg=bg_def, cursor="hand2",
                         bd=0, relief="flat")
        self._bg_def  = bg_def
        self._bg_hov  = bg_hov
        self._command = command

        self._lbl = tk.Label(self, text=text, bg=bg_def, fg=fg_col,
                              font=font, padx=padx, pady=pady,
                              width=width, cursor="hand2")
        self._lbl.pack()

        for w in (self, self._lbl):
            w.bind("<Button-1>", lambda e: self._command())
            w.bind("<Enter>",    lambda e: self._hover(True))
            w.bind("<Leave>",    lambda e: self._hover(False))

    def _hover(self, entering: bool):
        bg = self._bg_hov if entering else self._bg_def
        self.config(bg=bg)
        self._lbl.config(bg=bg)

    def set_appearance(self, text=None, bg_def=None, bg_hov=None, fg=None):
        """Update text and/or colour scheme and reset to idle state."""
        if text    is not None: self._lbl.config(text=text)
        if fg      is not None: self._lbl.config(fg=fg)
        if bg_def  is not None:
            self._bg_def = bg_def
            self.config(bg=bg_def)
            self._lbl.config(bg=bg_def)
        if bg_hov  is not None:
            self._bg_hov = bg_hov


# ── Main application ──────────────────────────────────────────────────────────

class MISHAApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MISHA  ·  LED Controller")
        self.configure(bg=BG)
        self.resizable(False, False)

        # ── state ──
        self._conn: serial.Serial | None = None
        self._connected   = False
        self._active_nm: int | None = None
        self._rx_running  = False

        # ── tk variables ──
        self._port_var      = tk.StringVar()
        self._intensity_var = tk.IntVar(value=50)
        self._status_text   = tk.StringVar(value="Disconnected")
        self._last_cmd_text = tk.StringVar(value="—")

        # ── panel status dots ──
        self._panel_a_dot: tk.Canvas | None = None
        self._panel_a_oval = None
        self._panel_b_dot: tk.Canvas | None = None
        self._panel_b_oval = None

        # ── wave-button registry ──
        self._wave_frames: dict[int, tk.Frame]  = {}
        self._wave_labels: dict[int, list]      = {}

        self._build_styles()
        self._build_ui()
        self._refresh_ports()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── ttk styles ────────────────────────────────────────────────────────────

    def _build_styles(self):
        st = ttk.Style(self)
        st.theme_use("clam")

        st.configure("Dark.TCombobox",
                      fieldbackground=BTN_DARK, background=BTN_DARK,
                      foreground=FG, selectbackground=CARD2,
                      arrowcolor=FG2, bordercolor=SEP,
                      lightcolor=SEP, darkcolor=SEP,
                      padding=(6, 4))
        st.map("Dark.TCombobox",
               fieldbackground=[("readonly", BTN_DARK)],
               foreground=[("readonly", FG)],
               selectbackground=[("readonly", CARD2)])

        st.configure("MISHA.Horizontal.TScale",
                      troughcolor=BTN_DARK, background=CARD,
                      sliderlength=18, troughrelief="flat",
                      sliderrelief="flat")
        st.map("MISHA.Horizontal.TScale",
               background=[("active", CARD)])

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(padx=24, pady=(18, 20))

        self._build_header(outer)
        self._build_sep(outer)
        self._build_connection(outer)
        self._build_sep(outer)
        self._build_wavelength_grid(outer)
        self._build_sep(outer)
        self._build_intensity(outer)
        self._build_sep(outer)
        self._build_actions(outer)
        self._build_sep(outer)
        self._build_serial_log(outer)
        self._build_footer(outer)

    # · Header ·

    def _build_header(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 14))

        tk.Label(row, text="MISHA", font=(FONT_FAMILY, 30, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(row, text=" LED Controller",
                 font=(FONT_FAMILY, 20), bg=BG, fg=FG).pack(side="left", pady=6)

        # status indicator (right-aligned)
        right = tk.Frame(row, bg=BG)
        right.pack(side="right")

        self._dot = tk.Canvas(right, width=10, height=10, bg=BG,
                              highlightthickness=0)
        self._dot_oval = self._dot.create_oval(1, 1, 9, 9, fill="#444466",
                                               outline="")
        self._dot.pack(side="right", padx=(4, 0), pady=6)

        tk.Label(right, textvariable=self._status_text,
                 font=(FONT_FAMILY, 10), bg=BG, fg=FG2).pack(side="right")

        # panel A / panel B active indicators
        tk.Label(right, text="B", font=(FONT_FAMILY, 9), bg=BG, fg=FG2).pack(side="right", padx=(10, 0))
        self._panel_b_dot = tk.Canvas(right, width=10, height=10, bg=BG, highlightthickness=0)
        self._panel_b_oval = self._panel_b_dot.create_oval(1, 1, 9, 9, fill="#444466", outline="")
        self._panel_b_dot.pack(side="right", padx=(4, 2), pady=6)

        tk.Label(right, text="A", font=(FONT_FAMILY, 9), bg=BG, fg=FG2).pack(side="right", padx=(10, 0))
        self._panel_a_dot = tk.Canvas(right, width=10, height=10, bg=BG, highlightthickness=0)
        self._panel_a_oval = self._panel_a_dot.create_oval(1, 1, 9, 9, fill="#444466", outline="")
        self._panel_a_dot.pack(side="right", padx=(4, 2), pady=6)

    # · Connection panel ·

    def _build_connection(self, parent):
        card = self._card(parent, "Connection")

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x")

        tk.Label(row, text="Port", font=(FONT_FAMILY, 11),
                 bg=CARD, fg=FG2, width=4, anchor="w").pack(side="left")

        self._port_combo = ttk.Combobox(row, textvariable=self._port_var,
                                         style="Dark.TCombobox",
                                         state="readonly", width=24,
                                         font=(FONT_FAMILY, 11))
        self._port_combo.pack(side="left", padx=(8, 8))

        self._btn(row, "⟳", self._refresh_ports,
                  width=3, font_size=14).pack(side="left", padx=(0, 10))

        self._connect_btn = self._btn(row, "Connect", self._toggle_connect,
                                       accent=True, width=10)
        self._connect_btn.pack(side="left")

    # · Wavelength grid ·

    def _build_wavelength_grid(self, parent):
        card = self._card(parent, "Wavelength")

        grid = tk.Frame(card, bg=CARD)
        grid.pack()

        for idx, nm in enumerate(WAVELENGTHS):
            col = idx % 8
            row = idx // 8
            fr = self._wave_tile(grid, nm)
            fr.grid(row=row, column=col, padx=5, pady=5)
            self._wave_frames[nm] = fr

    def _wave_tile(self, parent, nm: int) -> tk.Frame:
        """Build a single wavelength selector tile."""
        color = WAVE_COLOR[nm]

        fr = tk.Frame(parent, bg=BTN_DARK, cursor="hand2",
                      highlightbackground=BTN_DARK, highlightthickness=2,
                      relief="flat", bd=0)

        # colour swatch strip
        swatch = tk.Canvas(fr, width=56, height=6, bg=BTN_DARK,
                            highlightthickness=0)
        swatch.create_rectangle(0, 0, 56, 6, fill=color, outline="")
        swatch.pack(pady=(8, 3))

        # wavelength number
        lbl_nm = tk.Label(fr, text=str(nm),
                          font=(FONT_FAMILY, 13, "bold"),
                          bg=BTN_DARK, fg=FG)
        lbl_nm.pack()

        # band label
        if nm < 400:
            band = "UV"
        elif nm > 700:
            band = "IR"
        else:
            band = "nm"

        lbl_band = tk.Label(fr, text=band, font=(FONT_FAMILY, 8),
                             bg=BTN_DARK, fg=FG2)
        lbl_band.pack(pady=(1, 2))

        # physical panel indicator (A / A · B / A · B · C)
        lbl_panel = tk.Label(fr, text=PANEL_LEDS[nm], font=(FONT_FAMILY, 7),
                              bg=BTN_DARK, fg=FG2)
        lbl_panel.pack(pady=(0, 7))

        self._wave_labels[nm] = [swatch, lbl_nm, lbl_band, lbl_panel]

        # bind all children
        for w in (fr, swatch, lbl_nm, lbl_band, lbl_panel):
            w.bind("<Button-1>", lambda e, n=nm: self._toggle_wavelength(n))
            w.bind("<Enter>",    lambda e, n=nm: self._tile_hover(n, True))
            w.bind("<Leave>",    lambda e, n=nm: self._tile_hover(n, False))

        return fr

    # · Intensity panel ·

    def _build_intensity(self, parent):
        card = self._card(parent, "Intensity")

        slider_row = tk.Frame(card, bg=CARD)
        slider_row.pack(fill="x", pady=(0, 10))

        self._slider = ttk.Scale(slider_row, from_=0, to=100,
                                  orient="horizontal",
                                  variable=self._intensity_var,
                                  style="MISHA.Horizontal.TScale",
                                  length=380)
        self._slider.pack(side="left", padx=(0, 14))
        self._slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self._intensity_var.trace_add("write", self._on_intensity_trace)

        self._int_label = tk.Label(slider_row,
                                    text=f"{self._intensity_var.get()}%",
                                    font=(FONT_FAMILY, 18, "bold"),
                                    bg=CARD, fg=ACCENT, width=5, anchor="w")
        self._int_label.pack(side="left")

        # preset buttons
        preset_row = tk.Frame(card, bg=CARD)
        preset_row.pack(fill="x")

        for label, val in [("0%", 0), ("10%", 10), ("25%", 25),
                            ("50%", 50), ("75%", 75), ("100%", 100)]:
            self._btn(preset_row, label,
                      lambda v=val: self._set_intensity(v),
                      width=6, font_size=9).pack(side="left", padx=2)

    # · Action row ·

    def _build_actions(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(14, 6))

        self._off_btn = self._btn(row, "Turn Off All", self._turn_off_all,
                                   accent=False, width=14)
        self._off_btn.pack(side="left", padx=(0, 8))

        self._resend_btn = self._btn(row, "Resend", self._resend_last,
                                     accent=False, width=8)
        self._resend_btn.pack(side="left")

        # last command echo
        cmd_frame = tk.Frame(row, bg=BG)
        cmd_frame.pack(side="right")

        tk.Label(cmd_frame, text="Last cmd:", font=(FONT_FAMILY, 9),
                 bg=BG, fg=FG2).pack(side="left", padx=(0, 4))
        tk.Label(cmd_frame, textvariable=self._last_cmd_text,
                 font=(FONT_FAMILY, 9, "bold"), bg=BG, fg=FG2,
                 width=18, anchor="w").pack(side="left")

    # · Serial log ·

    def _build_serial_log(self, parent):
        card = self._card(parent, "Serial Log")

        self._log = tk.Text(card, height=4, bg=BTN_DARK, fg=FG2,
                             font=(FONT_FAMILY, 9), relief="flat", bd=0,
                             state="disabled", wrap="word",
                             insertbackground=FG, selectbackground=CARD2)
        self._log.pack(fill="x")

    def _append_log(self, line: str):
        """Append a line to the serial log (safe to call from any thread)."""
        self._log.config(state="normal")
        self._log.insert("end", line + "\n")
        # Keep at most 200 lines
        excess = int(self._log.index("end-1c").split(".")[0]) - 200
        if excess > 0:
            self._log.delete("1.0", f"{excess}.0")
        self._log.see("end")
        self._log.config(state="disabled")

    def _rx_thread(self):
        """Background thread: read lines from the Arduino and post to the log."""
        while self._rx_running and self._conn:
            try:
                raw = self._conn.readline()
                if raw:
                    line = raw.decode(errors="replace").strip()
                    if line:
                        self.after(0, self._append_log, line)
            except Exception:
                break

    # · Footer ·

    def _build_footer(self, parent):
        tk.Label(parent, text="Reed Terdal · MISHA v1.0",
                 font=(FONT_FAMILY, 9), bg=BG, fg=FG2).pack(pady=(10, 0))

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _card(self, parent, title: str) -> tk.Frame:
        """Return the inner content frame of a titled card section."""
        wrapper = tk.Frame(parent, bg=CARD, bd=0, relief="flat")
        wrapper.pack(fill="x", pady=(0, 0))

        hdr = tk.Frame(wrapper, bg=CARD)
        hdr.pack(fill="x", padx=14, pady=(10, 6))

        tk.Label(hdr, text=title.upper(),
                 font=(FONT_FAMILY, 9, "bold"),
                 bg=CARD, fg=FG2).pack(side="left")

        inner = tk.Frame(wrapper, bg=CARD)
        inner.pack(fill="x", padx=14, pady=(0, 14))
        return inner

    def _btn(self, parent, text, command,
             accent=False, width=10, font_size=11) -> _FlatButton:
        bg_def = ACCENT    if accent else BTN_DARK
        bg_hov = ACCENT_HV if accent else CARD2
        fg_col = "#FFFFFF"  if accent else FG
        return _FlatButton(parent, text, command,
                           bg_def=bg_def, bg_hov=bg_hov, fg_col=fg_col,
                           font=(FONT_FAMILY, font_size), width=width)

    def _build_sep(self, parent):
        tk.Frame(parent, bg=SEP, height=1).pack(fill="x", pady=10)

    # ── Port management ───────────────────────────────────────────────────────

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self._port_combo["values"] = ports
        if ports and not self._port_var.get():
            # Prefer USB serial adapters (Arduino typically shows up as usbserial/usbmodem)
            usb = next((p for p in ports if "usbserial" in p or "usbmodem" in p), None)
            self._port_var.set(usb or ports[0])

    def _toggle_connect(self):
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self._port_var.get()
        if not port:
            messagebox.showwarning("No Port",
                                   "Please select a serial port first.")
            return
        try:
            self._conn = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)           # allow Arduino to finish resetting
            self._conn.reset_input_buffer()   # discard reset noise
            self._connected  = True
            self._rx_running = True
            threading.Thread(target=self._rx_thread, daemon=True).start()
            self._connect_btn.set_appearance(
                text="Disconnect", bg_def=WARNING, bg_hov="#FFC040")
            self._set_status("Connected", SUCCESS)
        except serial.SerialException as exc:
            messagebox.showerror("Connection Error", str(exc))

    def _disconnect(self):
        self._rx_running = False    # signal RX thread to stop
        self._send_raw(0, 0)        # turn off board before closing
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._connected   = False
        self._active_nm   = None
        self._clear_selection()
        self._set_panel_status(False)
        self._connect_btn.set_appearance(
            text="Connect", bg_def=ACCENT, bg_hov=ACCENT_HV)
        self._set_status("Disconnected", "#444466")
        self._last_cmd_text.set("—")

    # ── LED control ───────────────────────────────────────────────────────────

    def _toggle_wavelength(self, nm: int):
        if not self._connected:
            messagebox.showinfo("Not Connected",
                                "Connect to the board before selecting an LED.")
            return
        if self._active_nm == nm:
            # second click → turn off
            self._turn_off_all()
        else:
            self._clear_selection()
            self._active_nm = nm
            self._set_tile_active(nm, True)
            self._send_raw(nm, self._intensity_var.get())
            self._set_panel_status(True)

    def _turn_off_all(self):
        self._clear_selection()
        self._set_panel_status(False)
        if self._connected:
            self._send_raw(0, 0)    # wavelength 0 maps to NUM_LEDS+1 → turnOffAll

    def _on_intensity_trace(self, *_):
        """Update the % label live while dragging — no serial command sent."""
        try:
            v = int(self._intensity_var.get())
            self._int_label.config(text=f"{v}%")
        except Exception:
            pass

    def _on_slider_release(self, _event):
        """Send the intensity command once when the user releases the slider."""
        v = int(self._intensity_var.get())
        if self._active_nm and self._connected:
            self._send_raw(self._active_nm, v)

    def _set_intensity(self, val: int):
        self._intensity_var.set(val)
        self._int_label.config(text=f"{val}%")
        if self._active_nm and self._connected:
            self._send_raw(self._active_nm, val)

    def _send_raw(self, wavelength: int, intensity: int):
        if not self._conn:
            return
        cmd = f"{wavelength},{intensity}\n"
        try:
            self._conn.write(cmd.encode())
            if wavelength == 0:
                self._last_cmd_text.set("—")
            else:
                self._last_cmd_text.set(f"{wavelength}nm @ {intensity}%  →  A · B")
        except serial.SerialException as exc:
            messagebox.showerror("Serial Error", str(exc))
            self._disconnect()

    # ── Tile highlight helpers ────────────────────────────────────────────────

    def _set_tile_active(self, nm: int, active: bool):
        color      = WAVE_COLOR[nm]
        tile_bg    = color    if active else BTN_DARK
        border_col = color    if active else BTN_DARK
        fg_main    = _contrast_text(color) if active else FG
        fg_band    = _contrast_text(color) if active else FG2

        fr = self._wave_frames[nm]
        fr.config(bg=tile_bg, highlightbackground=border_col,
                  highlightthickness=2)

        swatch, lbl_nm, lbl_band, lbl_panel = self._wave_labels[nm]
        swatch.config(bg=tile_bg)
        lbl_nm.config(bg=tile_bg, fg=fg_main)
        lbl_band.config(bg=tile_bg, fg=fg_band)
        lbl_panel.config(bg=tile_bg, fg=fg_band)

    def _tile_hover(self, nm: int, entering: bool):
        if nm == self._active_nm:
            return     # don't override active state
        bg = CARD2 if entering else BTN_DARK

        fr = self._wave_frames[nm]
        fr.config(bg=bg, highlightbackground=bg)
        swatch, lbl_nm, lbl_band, lbl_panel = self._wave_labels[nm]
        for w in (swatch, lbl_nm, lbl_band, lbl_panel):
            w.config(bg=bg)

    def _clear_selection(self):
        if self._active_nm is not None:
            self._set_tile_active(self._active_nm, False)
        self._active_nm = None

    # ── Status indicator ──────────────────────────────────────────────────────

    def _set_status(self, text: str, colour: str):
        self._status_text.set(text)
        self._dot.itemconfig(self._dot_oval, fill=colour)

    def _set_panel_status(self, active: bool):
        colour = SUCCESS if active else "#444466"
        if self._panel_a_dot:
            self._panel_a_dot.itemconfig(self._panel_a_oval, fill=colour)
        if self._panel_b_dot:
            self._panel_b_dot.itemconfig(self._panel_b_oval, fill=colour)

    def _resend_last(self):
        if self._active_nm and self._connected:
            self._send_raw(self._active_nm, self._intensity_var.get())

    # ── Window close ─────────────────────────────────────────────────────────

    def _on_close(self):
        if self._connected:
            self._disconnect()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = MISHAApp()
    app.mainloop()
