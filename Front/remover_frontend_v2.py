#!/usr/bin/env python3
"""
Frontend CustomTkinter para remoção de fundo/objeto — v2
Design System: identidade própria (transparência, precisão, camadas)
Paleta: teal/ciano cirúrgico + checkerboard real + tipografia intencional
"""

import sys
import types
import tkinter

# Shim de compatibilidade para ImageTk no Python 3.15+ (DEVE vir ANTES do import PIL)
try:
    from PIL import Image, ImageTk
except ImportError:
    from PIL import Image
    imgtk_mod = types.ModuleType('PIL.ImageTk')
    imgtk_mod.PhotoImage = tkinter.PhotoImage
    sys.modules['PIL.ImageTk'] = imgtk_mod

import io
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import customtkinter as ctk
from tkinter import Canvas, filedialog, messagebox


# ─── DESIGN TOKENS ───
SPACING = {
    "xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 20, "2xl": 24, "3xl": 32, "4xl": 40
}

RADIUS = {"sm": 6, "md": 10, "lg": 14, "xl": 18, "full": 9999}

ELEVATION = {
    "flat": {"border": 1, "border_color": "#1E293B"},
}

COLOR = {
    "bg": "#0A0F1A",
    "bg_secondary": "#111827",
    "bg_tertiary": "#1E293B",

    "primary": "#06D6A0",
    "primary_hover": "#05C090",

    "success": "#10B981",
    "success_hover": "#059669",
    "error": "#F87171",
    "error_hover": "#EF4444",
    "warning": "#FBBF24",

    "checker_light": "#1E293B",
    "checker_dark": "#111827",
    "checker_size": 20,

    "text": "#F0F4F8",
    "text_secondary": "#9BA8B8",
    "text_muted": "#6B7A8C",
    "text_on_primary": "#041512",

    "border": "#1E293B",
    "border_hover": "#2D3A4F",
}

FONT = {
    "display": ("Inter", 32, "bold"),
    "h3": ("Inter", 16, "bold"),
    "body_lg": ("Inter", 15, "normal"),
    "body": ("Inter", 13, "normal"),
    "caption": ("Inter", 11, "normal"),
    "caption_bold": ("Inter", 11, "bold"),
    "button": ("Inter", 13, "bold"),
    "button_lg": ("Inter", 15, "bold"),
    "mono": ("JetBrains Mono", 12, "normal"),
    "mono_sm": ("JetBrains Mono", 11, "normal"),
}


# ─── Theme Setup ───
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")



# ─── SVG Icons (external module) ───
from icons import ICONS



# ─── Backend real (ponte para ../pipeline.py; o back não é alterado) ───
from backend_bridge import (
    METODOS,
    ResultadoProcessamento,
    backend_processar,
    cancelar_backend,
)


# ─── UTILITÁRIOS ───
def font_spec(key: str) -> ctk.CTkFont:
    fam, sz, wt = FONT[key]
    return ctk.CTkFont(family=fam, size=sz, weight=wt)


# cairosvg é opcional: sem ele (ou sem a lib nativa do cairo) os ícones ficam transparentes.
# Fedora: pip install cairosvg && sudo dnf install cairo
try:
    import cairosvg
    HAS_CAIROSVG = True
except (ImportError, OSError):
    HAS_CAIROSVG = False


def svg_to_ctk_image(svg_str: str, size: tuple[int, int], color: Optional[str] = None) -> ctk.CTkImage:
    """Converte SVG string para CTkImage via PIL (placeholder transparente se não houver cairosvg)."""
    if not HAS_CAIROSVG or not svg_str:
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)

    if color:
        svg_str = svg_str.replace('stroke="currentColor"', f'stroke="{color}"')
        svg_str = svg_str.replace('fill="currentColor"', f'fill="{color}"')

    png_bytes = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=size[0],
        output_height=size[1],
    )
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return ctk.CTkImage(light_image=img, dark_image=img, size=size)


class IconCache:
    """Cache simples para CTkImages de SVG."""
    _cache: dict[str, ctk.CTkImage] = {}

    @classmethod
    def get(cls, name: str, size: tuple[int, int], color: Optional[str] = None) -> ctk.CTkImage:
        key = f"{name}_{size[0]}x{size[1]}_{color or 'default'}"
        if key not in cls._cache:
            cls._cache[key] = svg_to_ctk_image(ICONS[name], size, color)
        return cls._cache[key]


# ─── COMPONENTES ───
class AnimatedButton(ctk.CTkButton):
    """Button com feedback visual consistente: hover nativo do CTk + press customizado.
    Garante que cores e texto permaneçam corretos ao alternar disabled/normal."""
    def __init__(self, master, **kwargs):
        # Extrai cores antes de passar para super()
        self._original_fg = kwargs.get("fg_color", COLOR["primary"])
        self._hover_fg = kwargs.get("hover_color", COLOR["primary_hover"])
        self._press_fg = kwargs.pop("press_color", self._darken(self._original_fg))
        self._text_color = kwargs.get("text_color", COLOR["text_on_primary"])
        self._syncing = False  # guarda de recursão
        
        # Remove hover_color do kwargs para evitar conflito - usamos bind para press apenas
        kwargs.pop("hover_color", None)
        
        super().__init__(master, **kwargs)
        
        # Configura hover nativo do CTk (mais confiável)
        self._configure_hover()
        
        # Bind apenas para press/release (feedback tátil que CTk não tem nativamente)
        self.bind("<Button-1>", lambda e: self._on_press(), add="+")
        self.bind("<ButtonRelease-1>", lambda e: self._on_release(), add="+")

    @staticmethod
    def _darken(hex_color: str) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        factor = 0.8
        return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"

    def _configure_hover(self):
        """Configura hover_color nativo do CTk para transição suave automática."""
        try:
            self.configure(hover_color=self._hover_fg)
        except Exception:
            pass

    def _sync_colors(self):
        """Re-sincroniza cores quando widget é reconfigurado (ex: disabled→normal)."""
        if self._syncing:
            return
        if self.cget("state") != "disabled":
            self._syncing = True
            try:
                super().configure(fg_color=self._original_fg, text_color=self._text_color)
                self._configure_hover()
            finally:
                self._syncing = False

    def configure(self, **kwargs):
        """Override para interceptar mudança de estado e re-sincronizar cores."""
        state_change = "state" in kwargs and kwargs["state"] != "disabled"
        super().configure(**kwargs)
        if state_change:
            # Pequeno delay para garantir que CTk aplicou o estado
            self.after(10, self._sync_colors)

    def _on_press(self):
        if self.cget("state") != "disabled" and not self._syncing:
            super().configure(fg_color=self._press_fg)

    def _on_release(self):
        if self.cget("state") != "disabled" and not self._syncing:
            super().configure(fg_color=self._hover_fg)
            self.after(120, lambda: super().configure(fg_color=self._original_fg) if self.cget("state") != "disabled" and not self._syncing else None)

    def set_colors(self, fg: str, hover: str, text: str = None):
        self._original_fg = fg
        self._hover_fg = hover
        self._press_fg = self._darken(fg)
        if text:
            self._text_color = text
        self._sync_colors()


class IconButton(ctk.CTkButton):
    """Botão ícone compacto (remover, toggle)."""
    def __init__(self, master, icon_name: str = "close", size: int = 24, **kwargs):
        kwargs.setdefault("width", size)
        kwargs.setdefault("height", size)
        kwargs.setdefault("corner_radius", RADIUS["full"])
        kwargs.setdefault("fg_color", "transparent")
        kwargs.setdefault("hover_color", COLOR["bg_tertiary"])
        kwargs.setdefault("text", "")
        kwargs.setdefault("image", IconCache.get(icon_name, (size - 4, size - 4), COLOR["text_muted"]))
        super().__init__(master, **kwargs)


class CheckerboardCanvas(ctk.CTkFrame):
    """Canvas com padrão checkerboard desenhado — identidade visual do domínio.
    Encapsula o tk.Canvas interno; acesso externo só via métodos públicos."""
    def __init__(self, master, size: tuple[int, int] = (300, 300), **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.size = size
        self.checker_size = COLOR["checker_size"]
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._canvas: Optional[Canvas] = None
        self._criar_canvas()

    def _criar_canvas(self):
        self._canvas = Canvas(
            self,
            width=self.size[0],
            height=self.size[1],
            bg=COLOR["bg"],
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack()
        self._desenhar_checkerboard()

    def _desenhar_checkerboard(self):
        if not self._canvas:
            return
        w, h = self.size
        cs = self.checker_size
        light = COLOR["checker_light"]
        dark = COLOR["checker_dark"]

        for y in range(0, h, cs):
            for x in range(0, w, cs):
                color = light if ((x // cs) + (y // cs)) % 2 == 0 else dark
                self._canvas.create_rectangle(x, y, x + cs, y + cs, fill=color, outline="", tags="checker")

    def recolor_checkerboard(self, light: str, dark: str):
        """Recolore o padrão checkerboard (usado para estado sutil no DropZone vazio)."""
        if not self._canvas:
            return
        for item in self._canvas.find_withtag("checker"):
            # Alterna cores baseado na posição original
            coords = self._canvas.coords(item)
            if len(coords) >= 2:
                x, y = coords[0], coords[1]
                cs = self.checker_size
                color = light if ((int(x) // cs) + (int(y) // cs)) % 2 == 0 else dark
                self._canvas.itemconfig(item, fill=color)

    def mostrar_imagem(self, img: Image.Image):
        """Exibe imagem sobre o checkerboard (mostra transparência real)."""
        if not self._canvas:
            return
        self._canvas.delete("overlay")
        img_copy = img.convert("RGBA")
        img_copy.thumbnail(self.size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img_copy)
        x = (self.size[0] - img_copy.width) // 2
        y = (self.size[1] - img_copy.height) // 2
        self._canvas.create_image(x, y, anchor="nw", image=self._photo, tags="overlay")

    def limpar(self):
        if not self._canvas:
            return
        self._canvas.delete("overlay")
        self._photo = None


class DropZone(ctk.CTkFrame):
    """
    Área de upload com 2 estados: vazio (checkerboard sutil + ícone) / preenchido (thumbnail).
    Click em qualquer área do card vazio abre filedialog.
    """
    def __init__(self, master, on_change=None, on_remove=None, **kwargs):
        super().__init__(
            master,
            fg_color=COLOR["bg_secondary"],
            corner_radius=RADIUS["lg"],
            border_width=ELEVATION["flat"]["border"],
            border_color=ELEVATION["flat"]["border_color"],
            **kwargs,
        )
        self.grid_columnconfigure(0, weight=1)
        self.on_change = on_change
        self.on_remove = on_remove
        self.caminho_atual: Optional[Path] = None
        self._thumb_ctk = None
        self._criar_ui()
        self._estado_vazio()

    def _criar_ui(self):
        self.inner = ctk.CTkFrame(self, fg_color="transparent")
        self.inner.grid(row=0, column=0, sticky="nsew", padx=SPACING["xl"], pady=SPACING["xl"])
        self.inner.grid_columnconfigure(0, weight=1)

        # ── ESTADO VAZIO ──
        self.frame_vazio = ctk.CTkFrame(self.inner, fg_color="transparent")
        self.frame_vazio.grid_columnconfigure(0, weight=1)

        self.checker_bg = CheckerboardCanvas(self.frame_vazio, size=(200, 160))
        self.checker_bg.grid(row=0, column=0, pady=(SPACING["xl"], SPACING["md"]))
        self.checker_bg.recolor_checkerboard("#151d2e", "#0d131c")  # mais sutil

        self.icone_upload = ctk.CTkLabel(
            self.frame_vazio, text="",
            image=IconCache.get("upload", (48, 48), COLOR["primary"]),
            fg_color="transparent",
        )
        self.icone_upload.grid(row=0, column=0, pady=(SPACING["xl"], SPACING["md"]))

        self.titulo_vazio = ctk.CTkLabel(
            self.frame_vazio, text="Clique para selecionar sua imagem",
            font=font_spec("h3"), text_color=COLOR["text"],
        )
        self.titulo_vazio.grid(row=1, column=0, pady=(0, SPACING["xs"]))

        self.subtitulo_vazio = ctk.CTkLabel(
            self.frame_vazio, text="escolha um arquivo do seu computador",
            font=font_spec("body"), text_color=COLOR["text_secondary"],
        )
        self.subtitulo_vazio.grid(row=2, column=0, pady=(0, SPACING["md"]))

        self.formatos = ctk.CTkLabel(
            self.frame_vazio, text="PNG • JPG • WEBP • BMP  •  Até 50MB",
            font=font_spec("caption"), text_color=COLOR["text_muted"],
        )
        self.formatos.grid(row=3, column=0, pady=(0, SPACING["xl"]))

        # ── ESTADO PREENCHIDO ──
        self.frame_cheio = ctk.CTkFrame(self.inner, fg_color="transparent")
        self.frame_cheio.grid_columnconfigure(0, weight=1)

        self.thumb_container = ctk.CTkFrame(self.frame_cheio, fg_color=COLOR["bg"], corner_radius=RADIUS["md"])
        self.thumb_container.grid(row=0, column=0, pady=(0, SPACING["lg"]))

        self.lbl_thumb = ctk.CTkLabel(self.thumb_container, text="", width=140, height=140, corner_radius=RADIUS["md"])
        self.lbl_thumb.pack(padx=2, pady=2)

        self.info_frame = ctk.CTkFrame(self.frame_cheio, fg_color="transparent")
        self.info_frame.grid(row=1, column=0, pady=(0, SPACING["md"]))
        self.info_frame.grid_columnconfigure(0, weight=1)

        self.lbl_nome = ctk.CTkLabel(self.info_frame, text="", font=font_spec("body_lg"), text_color=COLOR["text"])
        self.lbl_nome.grid(row=0, column=0, pady=(0, SPACING["xs"]))

        self.lbl_detalhes = ctk.CTkLabel(self.info_frame, text="", font=font_spec("mono_sm"), text_color=COLOR["text_secondary"])
        self.lbl_detalhes.grid(row=1, column=0, pady=(0, SPACING["xs"]))

        self.lbl_dimensoes = ctk.CTkLabel(self.info_frame, text="", font=font_spec("mono_sm"), text_color=COLOR["text_muted"])
        self.lbl_dimensoes.grid(row=2, column=0)

        # Botão remover (topo direito)
        self.btn_remover = IconButton(self, icon_name="close", size=28, command=self._remover)

        # Click delegado no container vazio — único handler para toda a área
        self._bind_click_delegado(self.frame_vazio, self._on_click_vazio)

    def _estado_vazio(self):
        self.frame_cheio.grid_remove()
        self.frame_vazio.grid(row=0, column=0, sticky="nsew")
        self.btn_remover.place_forget()
        self.caminho_atual = None

    def _estado_cheio(self):
        self.frame_vazio.grid_remove()
        self.frame_cheio.grid(row=0, column=0, sticky="nsew")
        self.btn_remover.place(relx=1.0, y=SPACING["md"], anchor="ne", x=-SPACING["md"])

    def _abrir_seletor(self):
        if self.caminho_atual is None:
            caminho = filedialog.askopenfilename(
                title="Selecionar imagem",
                filetypes=[("Imagens", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Todos", "*.*")],
            )
            if caminho:
                self.definir_arquivo(Path(caminho))

    def definir_arquivo(self, caminho: Path):
        self.caminho_atual = caminho
        dimensoes = ""
        try:
            img = Image.open(caminho)
            dimensoes = f"{img.width} × {img.height} px"
            img = img.convert("RGBA")
            img.thumbnail((136, 136), Image.Resampling.LANCZOS)
            self._thumb_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            self.lbl_thumb.configure(image=self._thumb_ctk, text="")
        except Exception:
            self._thumb_ctk = None
            self.lbl_thumb.configure(image="", text="🖼️", font=ctk.CTkFont(size=40))

        tamanho_mb = caminho.stat().st_size / (1024 * 1024)
        self.lbl_nome.configure(text=caminho.name)
        self.lbl_detalhes.configure(text=f"{caminho.suffix.upper()[1:]}  •  {tamanho_mb:.1f} MB")
        self.lbl_dimensoes.configure(text=dimensoes)
        self._estado_cheio()
        if self.on_change:
            self.on_change(caminho)

    def _remover(self):
        self._thumb_ctk = None
        self.lbl_thumb.configure(image="", text="")
        self._estado_vazio()
        if self.on_remove:
            self.on_remove()

    def limpar(self):
        self._remover()

    # ── Helpers de click delegado ──
    def _bind_click_delegado(self, widget, handler):
        """Vincula click no widget e propaga para filhos recursivamente (uma única vez)."""
        widget.bind("<Button-1>", handler, add="+")
        for child in widget.winfo_children():
            self._bind_click_delegado(child, handler)

    def _on_click_vazio(self, event):
        """Handler único: abre seletor só se estiver vazio e click não foi no botão remover."""
        if self.caminho_atual is None:
            # Evita disparar se click originou no btn_remover (que fica no parent, não no frame_vazio)
            if event.widget != self.btn_remover:
                self._abrir_seletor()


class SegmentedToggle(ctk.CTkFrame):
    """Toggle Fundo/Objeto — ícones SVG, labels curtos."""
    def __init__(self, master, on_change=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_change = on_change
        self.valor = "fundo"
        self._criar_ui()

    def _criar_ui(self):
        self.container = ctk.CTkFrame(self, fg_color=COLOR["bg_tertiary"], corner_radius=RADIUS["md"])
        self.container.pack(fill="x")
        self.container.grid_columnconfigure((0, 1), weight=1, uniform="toggle")

        self.btn_fundo = ctk.CTkButton(
            self.container, text="  Fundo", font=font_spec("button"),
            fg_color=COLOR["primary"], hover_color=COLOR["primary_hover"],
            text_color=COLOR["text_on_primary"], corner_radius=RADIUS["sm"],
            height=44, border_width=0, command=lambda: self._set("fundo"),
            image=IconCache.get("layers", (18, 18), COLOR["text_on_primary"]),
            compound="left", anchor="center",
        )
        self.btn_fundo.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        self.btn_objeto = ctk.CTkButton(
            self.container, text="  Objeto", font=font_spec("button"),
            fg_color="transparent", hover_color=COLOR["bg_secondary"],
            text_color=COLOR["text_secondary"], corner_radius=RADIUS["sm"],
            height=44, border_width=0, command=lambda: self._set("objeto"),
            image=IconCache.get("target", (18, 18), COLOR["text_secondary"]),
            compound="left", anchor="center",
        )
        self.btn_objeto.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

    def _set(self, valor: str):
        self.valor = valor
        if valor == "fundo":
            self._config_btn(self.btn_fundo, active=True, icon="layers")
            self._config_btn(self.btn_objeto, active=False, icon="target")
        else:
            self._config_btn(self.btn_objeto, active=True, icon="target")
            self._config_btn(self.btn_fundo, active=False, icon="layers")
        if self.on_change:
            self.on_change(valor)

    def _config_btn(self, btn: ctk.CTkButton, active: bool, icon: str):
        """Configura visual do botão do toggle."""
        if active:
            btn.configure(
                fg_color=COLOR["primary"], hover_color=COLOR["primary_hover"],
                text_color=COLOR["text_on_primary"],
                image=IconCache.get(icon, (18, 18), COLOR["text_on_primary"]),
            )
        else:
            btn.configure(
                fg_color="transparent", hover_color=COLOR["bg_secondary"],
                text_color=COLOR["text_secondary"],
                image=IconCache.get(icon, (18, 18), COLOR["text_secondary"]),
            )


class CollapsiblePanel(ctk.CTkFrame):
    """Painel colapsável de configurações avançadas."""
    def __init__(self, master, titulo="Configurações avançadas", **kwargs):
        super().__init__(
            master, fg_color=COLOR["bg_secondary"], corner_radius=RADIUS["lg"],
            border_width=ELEVATION["flat"]["border"], border_color=ELEVATION["flat"]["border_color"],
            **kwargs,
        )
        self.expanded = False
        self.titulo = titulo
        self._criar_ui()

    def _criar_ui(self):
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=SPACING["lg"], pady=(SPACING["lg"], 0))
        self.header.grid_columnconfigure(0, weight=1)

        self.btn_toggle = ctk.CTkButton(
            self.header, text=f"  {self.titulo}", font=font_spec("body"),
            fg_color="transparent", text_color=COLOR["text"], anchor="w", height=40,
            hover_color=COLOR["bg_tertiary"],
            command=self._alternar,
            image=IconCache.get("settings", (18, 18), COLOR["text_secondary"]),
            compound="left",
        )
        self.btn_toggle.grid(row=0, column=0, sticky="ew")

        self.chevron = ctk.CTkLabel(
            self.header, text="",
            image=IconCache.get("chevron_down", (16, 16), COLOR["text_muted"]),
            fg_color="transparent",
        )
        self.chevron.grid(row=0, column=1, padx=(0, SPACING["md"]))

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid_columnconfigure(0, weight=1)

        # Contador de linha automático para sliders
        self._slider_row = 0
        self._add_slider(
            label="Qualidade do resultado",
            hint="Valores maiores = melhor qualidade, processamento mais lento",
            from_=10, to=100, steps=90, default=85,
            format_val=lambda v: f"{int(v)}%",
            attr_slider="slider_qualidade", attr_label="lbl_qualidade",
        )

        sep = ctk.CTkFrame(self.content, height=1, fg_color=COLOR["border"])
        sep.grid(row=self._slider_row, column=0, sticky="ew", padx=SPACING["lg"], pady=SPACING["md"])
        self._slider_row += 1

        self._add_slider(
            label="Sensibilidade da detecção",
            hint="Baixa = detecta menos (conservador)  •  Alta = detecta mais (agressivo)",
            from_=0.1, to=1.0, steps=90, default=0.5,
            format_val=self._fmt_sensibilidade,
            attr_slider="slider_sensibilidade", attr_label="lbl_sensibilidade",
        )

        sep2 = ctk.CTkFrame(self.content, height=1, fg_color=COLOR["border"])
        sep2.grid(row=self._slider_row, column=0, sticky="ew", padx=SPACING["lg"], pady=SPACING["md"])
        self._slider_row += 1
        self._add_campos_backend()

    def _add_campos_backend(self):
        """Campos que o back precisa: qual objeto segmentar e qual método usar."""
        row = self._slider_row
        ctk.CTkLabel(self.content, text="Objeto", font=font_spec("body_lg"),
                     text_color=COLOR["text"], anchor="w").grid(
            row=row, column=0, sticky="ew", padx=SPACING["lg"], pady=(SPACING["md"], 0))
        ctk.CTkLabel(self.content, text="Ex.: cachorro, gato, pessoa, carro (vazio = qualquer objeto). Só vale para IA.",
                     font=font_spec("caption"), text_color=COLOR["text_muted"], anchor="w").grid(
            row=row + 1, column=0, sticky="ew", padx=SPACING["lg"], pady=(SPACING["xs"], SPACING["xs"]))
        self.entry_objeto = ctk.CTkEntry(
            self.content, placeholder_text="animal", font=font_spec("body"),
            fg_color=COLOR["bg_tertiary"], border_color=COLOR["border_hover"],
            text_color=COLOR["text"], height=36,
        )
        self.entry_objeto.insert(0, "animal")
        self.entry_objeto.grid(row=row + 2, column=0, sticky="ew", padx=SPACING["lg"])

        ctk.CTkLabel(self.content, text="Método de segmentação", font=font_spec("body_lg"),
                     text_color=COLOR["text"], anchor="w").grid(
            row=row + 3, column=0, sticky="ew", padx=SPACING["lg"], pady=(SPACING["md"], SPACING["xs"]))
        self.menu_metodo = ctk.CTkOptionMenu(
            self.content, values=list(METODOS.keys()), font=font_spec("body"),
            fg_color=COLOR["bg_tertiary"], button_color=COLOR["primary"],
            button_hover_color=COLOR["primary_hover"], text_color=COLOR["text"],
            height=36,
        )
        self.menu_metodo.grid(row=row + 4, column=0, sticky="ew", padx=SPACING["lg"])
        self._slider_row += 5

    def _add_slider(self, parent=None, label="", hint="", from_=0, to=100, steps=10, default=50,
                    format_val=str, attr_slider="", attr_label=""):
        """Adiciona slider com row automático."""
        if parent is None:
            parent = self.content
        row = self._slider_row

        lbl_frame = ctk.CTkFrame(parent, fg_color="transparent")
        lbl_frame.grid(row=row, column=0, sticky="ew", padx=SPACING["lg"],
                       pady=(SPACING["lg"] if row == 0 else SPACING["md"], SPACING["xs"]))
        lbl_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(lbl_frame, text=label, font=font_spec("body_lg"),
                     text_color=COLOR["text"], anchor="w").grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(lbl_frame, text=hint, font=font_spec("caption"),
                     text_color=COLOR["text_muted"], anchor="w").grid(row=1, column=0, sticky="ew", pady=(SPACING["xs"], 0))

        slider = ctk.CTkSlider(
            parent, from_=from_, to=to, number_of_steps=steps,
            button_color=COLOR["primary"], progress_color=COLOR["primary"],
            button_hover_color=COLOR["primary_hover"],
        )
        slider.set(default)
        slider.grid(row=row + 1, column=0, sticky="ew", padx=SPACING["lg"], pady=(0, SPACING["xs"]))
        setattr(self, attr_slider, slider)

        lbl_val = ctk.CTkLabel(parent, text=format_val(default), font=font_spec("caption_bold"),
                               text_color=COLOR["primary"], anchor="e")
        lbl_val.grid(row=row + 2, column=0, sticky="ew", padx=SPACING["lg"])
        setattr(self, attr_label, lbl_val)

        slider.configure(command=lambda v: lbl_val.configure(text=format_val(v)))

        self._slider_row += 3

    @staticmethod
    def _fmt_sensibilidade(v: float) -> str:
        if v < 0.35:
            return f"Baixa ({v:.2f}) — detecta menos"
        if v > 0.65:
            return f"Alta ({v:.2f}) — detecta mais"
        return f"Equilibrado ({v:.2f})"

    def _alternar(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.content.pack(fill="x", padx=0, pady=(0, SPACING["lg"]))
            self.chevron.configure(image=IconCache.get("chevron_up", (16, 16), COLOR["text_muted"]))
        else:
            self.content.pack_forget()
            self.chevron.configure(image=IconCache.get("chevron_down", (16, 16), COLOR["text_muted"]))

    def obter_valores(self) -> tuple[int, float]:
        return int(self.slider_qualidade.get()), float(self.slider_sensibilidade.get())

    def obter_opcoes(self) -> tuple[str, str]:
        """(objeto, método) no formato que o back espera."""
        return self.entry_objeto.get().strip(), METODOS[self.menu_metodo.get()]

    def definir_estado(self, estado: str):
        self.entry_objeto.configure(state=estado)
        self.menu_metodo.configure(state=estado)
        self.slider_qualidade.configure(state=estado)
        self.slider_sensibilidade.configure(state=estado)


class ProgressBar(ctk.CTkFrame):
    """Barra de progresso linear como OVERLAY no rodapé da janela — só aparece quando ativa."""
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=COLOR["bg_secondary"], corner_radius=RADIUS["md"],
                         border_width=1, border_color=COLOR["border"])
        self._visivel = False

        # Container interno com padding
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="x", padx=SPACING["lg"], pady=SPACING["md"])
        inner.grid_columnconfigure(0, weight=1)

        self.bar = ctk.CTkProgressBar(inner, height=4, corner_radius=2,
                                      progress_color=COLOR["primary"], fg_color=COLOR["bg_tertiary"])
        self.bar.set(0)
        self.bar.grid(row=0, column=0, sticky="ew", pady=(0, SPACING["xs"]))

        self.lbl = ctk.CTkLabel(inner, text="", font=font_spec("caption"),
                                text_color=COLOR["text_secondary"], anchor="w")
        self.lbl.grid(row=1, column=0, sticky="ew")

    def mostrar(self):
        if not self._visivel:
            # Posiciona como overlay no rodapé da janela principal
            self.place(relx=0.5, rely=1.0, anchor="s", relwidth=0.92, y=-SPACING["lg"])
            self._visivel = True
            self.lift()  # Garante que fique sobre outros elementos

    def esconder(self):
        if self._visivel:
            self.place_forget()
            self._visivel = False

    def atualizar(self, pct: int, msg: str):
        self.mostrar()
        self.bar.set(pct / 100)
        self.lbl.configure(text=msg, text_color=COLOR["text_secondary"])

    def sucesso(self, tempo: float, detalhe: str = ""):
        self.mostrar()
        self.bar.set(1)
        texto = f"Concluído em {tempo:.1f}s" + (f"  •  {detalhe}" if detalhe else "")
        self.lbl.configure(text=texto, text_color=COLOR["success"])
        # Auto-esconder após 3s
        self.after(3000, self.esconder)

    def erro(self, msg: str):
        self.mostrar()
        self.bar.set(0)
        self.lbl.configure(text=f"Erro: {msg}", text_color=COLOR["error"])
        # Auto-esconder após 5s
        self.after(5000, self.esconder)

    def resetar(self):
        self.bar.set(0)
        self.lbl.configure(text="")
        self.esconder()


class PreviewPanel(ctk.CTkFrame):
    """Painel de preview com checkerboard real — mostra transparência do resultado."""
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._criar_ui()

    def _criar_ui(self):
        self.card = ctk.CTkFrame(self, fg_color=COLOR["bg_secondary"], corner_radius=RADIUS["lg"],
                                 border_width=ELEVATION["flat"]["border"], border_color=COLOR["border"])
        self.card.grid(row=0, column=0, sticky="nsew")
        self.card.grid_columnconfigure(0, weight=1)
        self.card.grid_rowconfigure(0, weight=1)

        self.checkerboard = CheckerboardCanvas(self.card, size=(360, 360))
        self.checkerboard.grid(row=0, column=0, padx=SPACING["lg"], pady=SPACING["lg"])

        self.placeholder = ctk.CTkLabel(
            self.card,
            text="O resultado aparecerá aqui\n(transparência sobre checkerboard)",
            font=font_spec("body"), text_color=COLOR["text_muted"],
            justify="center",
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def mostrar_resultado(self, img: Image.Image):
        self.placeholder.place_forget()
        self.checkerboard.mostrar_imagem(img)

    def resetar(self):
        self.checkerboard.limpar()
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")


class HeroFrame(ctk.CTkFrame):
    """Hero fixo no topo — identidade imediata, não some no scroll."""
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self._criar_ui()
        self._animar_entrada()

    def _criar_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=0, column=0, sticky="ew", padx=SPACING["2xl"], pady=(SPACING["2xl"], SPACING["lg"]))
        container.grid_columnconfigure(0, weight=0)  # ícone NÃO expande
        container.grid_columnconfigure(1, weight=1)  # título expande

        # Ícone vetorial consistente (layers = camadas/recorte)
        self.icone = ctk.CTkLabel(
            container, text="",
            image=IconCache.get("layers", (28, 28), COLOR["primary"]),
            fg_color="transparent"
        )
        self.icone.grid(row=0, column=0, sticky="w", padx=(0, SPACING["sm"]))

        self.titulo = ctk.CTkLabel(
            container,
            text="Remoção de Fundo / Objeto",
            font=font_spec("display"),
            text_color=COLOR["text"],
            anchor="w",
        )
        self.titulo.grid(row=0, column=1, sticky="w")

        self.subtitle = ctk.CTkLabel(
            container,
            text="Selecione uma imagem → veja o fundo sumir em segundos",
            font=font_spec("body_lg"),
            text_color=COLOR["text_secondary"],
            anchor="w",
        )
        self.subtitle.grid(row=1, column=1, sticky="w", pady=(SPACING["sm"], 0))

        self.divider = ctk.CTkFrame(self, height=1, fg_color=COLOR["border"])
        self.divider.grid(row=1, column=0, sticky="ew", padx=SPACING["2xl"], pady=(0, SPACING["md"]))

    def _animar_entrada(self):
        self.titulo.configure(text_color=COLOR["text_muted"])
        self.subtitle.configure(text_color=COLOR["text_muted"])
        self.divider.configure(fg_color=COLOR["bg"])

        self.after(50, lambda: self.titulo.configure(text_color=COLOR["text"]))
        self.after(150, lambda: self.subtitle.configure(text_color=COLOR["text_secondary"]))
        self.after(250, lambda: self.divider.configure(fg_color=COLOR["border"]))


# ─── APP PRINCIPAL ───
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Remoção de Fundo / Objeto")
        self.geometry("720x820")
        self.minsize(600, 700)
        self.configure(fg_color=COLOR["bg"])

        # Grid simples: hero fixo (row=0), main scrollable preenche o resto (row=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.fila: queue.Queue = queue.Queue()
        self.thread_proc: Optional[threading.Thread] = None
        self.resultado: Optional[ResultadoProcessamento] = None

        self._criar_ui()
        self._poll_queue()

    def _criar_ui(self):
        # ── HERO (fixo no topo) ──
        self.hero = HeroFrame(self)
        self.hero.grid(row=0, column=0, sticky="ew")

        # ── MAIN SCROLLABLE (preenche espaço restante) ──
        main = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        main.grid(row=1, column=0, sticky="nsew", padx=SPACING["2xl"], pady=SPACING["md"])
        main.grid_columnconfigure(0, weight=1)

        # 1. Drop Zone
        self.dropzone = DropZone(main, on_change=self._on_arquivo, on_remove=self._on_remover)
        self.dropzone.grid(row=0, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 2. Toggle Modo
        self.toggle = SegmentedToggle(main, on_change=self._on_modo)
        self.toggle.grid(row=1, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 3. Painel Avançado
        self.painel = CollapsiblePanel(main)
        self.painel.grid(row=2, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 4. Botão Processar
        self.btn_proc = AnimatedButton(
            main, text="  Processar Imagem", font=font_spec("button_lg"),
            fg_color=COLOR["primary"], hover_color=COLOR["primary_hover"],
            text_color=COLOR["text_on_primary"], corner_radius=RADIUS["md"],
            height=52, state="disabled", command=self._iniciar,
            image=IconCache.get("sparkle", (20, 20), COLOR["text_on_primary"]),
            compound="left", anchor="center",
        )
        self.btn_proc.grid(row=3, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 5. Preview Panel
        self.preview = PreviewPanel(main, height=420)
        self.preview.grid(row=4, column=0, sticky="ew", pady=(0, SPACING["xl"]))
        self.preview.grid_propagate(False)

        # 6. Botão Salvar (aparece só quando concluir)
        self.btn_salvar = AnimatedButton(
            main, text="  Salvar Resultado", font=font_spec("button_lg"),
            fg_color=COLOR["success"], hover_color=COLOR["success_hover"],
            text_color=COLOR["text_on_primary"], corner_radius=RADIUS["md"],
            height=48, state="disabled", command=self._salvar,
            image=IconCache.get("download", (20, 20), COLOR["text_on_primary"]),
            compound="left", anchor="center",
        )

        # ── PROGRESS BAR COMO OVERLAY (não no grid, posicionada via place()) ──
        self.progress = ProgressBar(self)

    # ── CALLBACKS ──
    def _reset_state(self, habilitar_processar: bool = False):
        """Reseta estado comum: limpa preview, progresso, resultado, btn_salvar.
        habilitar_processar=True quando arquivo selecionado; False quando removido."""
        self.btn_proc.configure(state="normal" if habilitar_processar else "disabled")
        self.btn_salvar.configure(state="disabled")
        self.btn_salvar.grid_remove()
        self.progress.resetar()
        self.resultado = None
        self.preview.resetar()

    def _on_arquivo(self, caminho: Path):
        self._reset_state(habilitar_processar=True)

    def _on_remover(self):
        self._reset_state(habilitar_processar=False)

    def _on_modo(self, modo: str):
        pass  # o backend recebe o modo no processamento

    def _set_controles(self, ativo: bool):
        estado = "normal" if ativo else "disabled"
        self.painel.btn_toggle.configure(state=estado)
        self.painel.definir_estado(estado)
        self.toggle.btn_fundo.configure(state=estado)
        self.toggle.btn_objeto.configure(state=estado)

    def _iniciar(self):
        if not self.dropzone.caminho_atual:
            return

        self.btn_proc.configure(state="disabled", text="  Processando...")
        self._set_controles(False)
        self.progress.resetar()
        self.progress.mostrar()

        qualidade, sensibilidade = self.painel.obter_valores()
        objeto, metodo = self.painel.obter_opcoes()
        modo = self.toggle.valor

        self.thread_proc = threading.Thread(
            target=self._worker,
            args=(str(self.dropzone.caminho_atual), modo, qualidade, sensibilidade, objeto, metodo),
            daemon=True,
        )
        self.thread_proc.start()

    def _worker(self, entrada, modo, qualidade, sensibilidade, objeto, metodo):
        try:
            def cb(pct, msg):
                self.fila.put(("prog", pct, msg))
            res = backend_processar(entrada, modo, qualidade, sensibilidade, cb,
                                    objeto=objeto, metodo=metodo)
            self.fila.put(("ok", res))
        except Exception as e:
            self.fila.put(("err", str(e)))

    def _poll_queue(self):
        try:
            while True:
                tipo, *dados = self.fila.get_nowait()
                try:
                    if tipo == "prog":
                        self.progress.atualizar(*dados)
                    elif tipo == "ok":
                        self._finalizar_ok(dados[0])
                    elif tipo == "err":
                        self._finalizar_err(dados[0])
                except Exception as e:
                    # Se algo falhar ao finalizar, o polling continua rodando
                    self._finalizar_err(str(e))
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _finalizar_comum(self):
        """Estado comum pós-processamento (sucesso ou erro)."""
        self.btn_proc.configure(state="normal", text="  Processar Imagem")
        self._set_controles(True)

    def _finalizar_ok(self, res: ResultadoProcessamento):
        self.resultado = res
        self.progress.sucesso(res.tempo_segundos, res.mensagem)
        self.preview.mostrar_resultado(res.imagem)
        self._finalizar_comum()
        self.btn_salvar.grid(row=5, column=0, sticky="ew", pady=(SPACING["md"], 0))
        self.btn_salvar.configure(state="normal")

    def _finalizar_err(self, err: str):
        self.progress.erro(err)
        self._finalizar_comum()
        messagebox.showerror("Erro no processamento", err)

    def _salvar(self):
        if not self.resultado:
            return
        caminho = filedialog.asksaveasfilename(
            title="Salvar resultado", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WEBP", "*.webp"), ("Todos", "*.*")],
            initialfile=f"{self.dropzone.caminho_atual.stem}_processado.png",
        )
        if caminho:
            try:
                img = self.resultado.imagem
                # JPEG não suporta transparência
                if caminho.lower().endswith((".jpg", ".jpeg")) and img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(caminho)
                messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho}")
            except Exception as e:
                messagebox.showerror("Erro ao salvar", str(e))


if __name__ == "__main__":
    App().mainloop()