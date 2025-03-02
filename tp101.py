import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class Place:
    def __init__(self, name, tokens=0):
        self.name = name
        self.tokens = tokens

    def __repr__(self):
        return f"Place({self.name}, tokens={self.tokens})"


class Transition:
    def __init__(self, name):
        self.name = name
        self.input_arcs = {}
        self.output_arcs = {}

    def add_input_arc(self, place, weight):
        self.input_arcs[place.name] = weight

    def add_output_arc(self, place, weight):
        self.output_arcs[place.name] = weight

    def is_enabled(self, places):
        return all(places[name].tokens >= weight for name, weight in self.input_arcs.items())


class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}
        self.arcs = []
        self.pos = {}

    def add_place(self, name, tokens=0):
        self.places[name] = Place(name, tokens)
        self.pos[name] = np.random.rand(2) * 2 - 1

    def add_transition(self, name):
        self.transitions[name] = Transition(name)
        self.pos[name] = np.random.rand(2) * 2 - 1

    def add_arc(self, source, target, weight=1):
        self.arcs.append((source, target, weight))
        if source in self.places and target in self.transitions:
            self.transitions[target].add_input_arc(self.places[source], weight)
        elif source in self.transitions and target in self.places:
            self.transitions[source].add_output_arc(self.places[target], weight)

    def get_networkx_graph(self):
        G = nx.DiGraph()
        for node, pos in self.pos.items():
            G.add_node(node, pos=pos)
        for src, dst, weight in self.arcs:
            G.add_edge(src, dst, weight=weight)
        return G

    def is_bounded(self):
        return all(p.tokens >= 0 for p in self.places.values())

    def has_live_transitions(self):
        return any(t.is_enabled(self.places) for t in self.transitions.values())


class LightTheme:
    # Main color scheme (white and blue gradients)
    PRIMARY = "#ffffff"
    SECONDARY = "#f8f9fa"
    TERTIARY = "#e9ecef"

    # Blue gradients
    ACCENT_LIGHT = "#4895ef"
    ACCENT_MEDIUM = "#3f72af"
    ACCENT_DARK = "#112d4e"

    # Node colors
    PLACE_COLOR = "#4895ef"
    TRANSITION_COLOR = "#3f72af"

    # Text and border colors
    TEXT_COLOR = "#212529"
    TEXT_SECONDARY = "#495057"
    BORDER_COLOR = "#dee2e6"

    # Button styles
    BUTTON_BG = "#4895ef"
    BUTTON_HOVER = "#3a83d8"
    BUTTON_TEXT = "#ffffff"

    # Entry field styles
    ENTRY_BG = "#f8f9fa"
    ENTRY_BORDER = "#ced4da"

    # Graph colors
    GRAPH_BG = "#f8f9fa"
    EDGE_COLOR = "#adb5bd"

    # Modern UI parameters
    CORNER_RADIUS = 8
    BUTTON_STYLE = {
        "font": ("Segoe UI", 10),
        "borderwidth": 0,
        "relief": tk.FLAT,
        "padx": 12,
        "pady": 6
    }


class PetriNetApp:
    def __init__(self, root):
        self.root = root
        self.petri_net = PetriNet()
        self.dragging = None
        self.sidebar_visible = True
        self.sidebar_width = 350
        self.setup_ui()
        self.setup_graph()

    def setup_ui(self):
        self.root.title("PetriNet Studio")
        self.root.geometry("1600x900")
        self.root.configure(bg=LightTheme.PRIMARY)

        # Create main frame with horizontal layout
        self.main_frame = tk.Frame(self.root, bg=LightTheme.PRIMARY)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create collapsible sidebar
        self.sidebar_frame = tk.Frame(self.main_frame, bg=LightTheme.SECONDARY, width=self.sidebar_width)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.sidebar_frame.pack_propagate(False)  # Prevent the frame from shrinking

        # Create visualization area
        self.visualization_container = tk.Frame(self.main_frame, bg=LightTheme.PRIMARY)
        self.visualization_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Toggle button for sidebar
        self.toggle_btn_frame = tk.Frame(self.visualization_container, bg=LightTheme.PRIMARY, height=40)
        self.toggle_btn_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.toggle_btn = tk.Button(self.toggle_btn_frame, text="≡ Hide Panel", command=self.toggle_sidebar,
                                    bg=LightTheme.PRIMARY, fg=LightTheme.TEXT_COLOR,
                                    relief=tk.FLAT, borderwidth=0, padx=15, pady=5,
                                    font=("Segoe UI", 10), anchor="w")
        self.toggle_btn.pack(side=tk.LEFT)

        # Visualization frame
        self.visualization_frame = tk.Frame(self.visualization_container, bg=LightTheme.PRIMARY)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create canvas for network visualization
        self.figure = plt.figure(figsize=(10, 8), facecolor=LightTheme.GRAPH_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Sidebar content
        self.setup_sidebar()

    def setup_sidebar(self):
        # App title with gradient effect
        header_frame = tk.Frame(self.sidebar_frame, bg=LightTheme.SECONDARY, height=80)
        header_frame.pack(fill=tk.X)

        # Blue gradient line under the header
        gradient_line = tk.Canvas(header_frame, height=3, bg=LightTheme.SECONDARY,
                                  highlightthickness=0)
        gradient_line.pack(fill=tk.X, side=tk.BOTTOM)
        gradient_line.create_rectangle(0, 0, self.sidebar_width, 3,
                                       fill=LightTheme.ACCENT_MEDIUM, outline="")

        # Title
        tk.Label(header_frame, text="PetriNet Studio", font=("Segoe UI", 18, "bold"),
                 fg=LightTheme.ACCENT_DARK, bg=LightTheme.SECONDARY).pack(pady=20)

        # Network definition section
        definition_frame = tk.LabelFrame(self.sidebar_frame, text="Network Definition",
                                         font=("Segoe UI", 11),
                                         fg=LightTheme.TEXT_SECONDARY,
                                         bg=LightTheme.SECONDARY,
                                         bd=1, highlightbackground=LightTheme.BORDER_COLOR)
        definition_frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        # Input fields
        input_fields = [
            ("Places", 'places_entry'),
            ("Transitions", 'transitions_entry'),
            ("Arcs", 'arcs_entry'),
            ("Markings", 'marking_entry')
        ]

        for text, var_name in input_fields:
            frame = tk.Frame(definition_frame, bg=LightTheme.SECONDARY)
            frame.pack(fill=tk.X, pady=8, padx=10)

            tk.Label(frame, text=text, font=("Segoe UI", 10),
                     fg=LightTheme.TEXT_SECONDARY, bg=LightTheme.SECONDARY,
                     width=10, anchor='w').pack(side=tk.LEFT, padx=(0, 10))

            entry = tk.Entry(frame, bg=LightTheme.ENTRY_BG, fg=LightTheme.TEXT_COLOR,
                             insertbackground=LightTheme.TEXT_COLOR,
                             relief=tk.SOLID, bd=1, highlightthickness=0)
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            entry.config(highlightbackground=LightTheme.ENTRY_BORDER)
            setattr(self, var_name, entry)

        # Actions section
        actions_frame = tk.LabelFrame(self.sidebar_frame, text="Actions",
                                      font=("Segoe UI", 11),
                                      fg=LightTheme.TEXT_SECONDARY,
                                      bg=LightTheme.SECONDARY,
                                      bd=1, highlightbackground=LightTheme.BORDER_COLOR)
        actions_frame.pack(fill=tk.X, padx=15, pady=(15, 5))

        # Action buttons
        actions = [
            ("Load Model", self.load_net_file),
            ("Generate", self.generate_network),
            ("Analyze", self.analyze_net),
            ("Reset", self.reset_model)
        ]

        buttons_frame = tk.Frame(actions_frame, bg=LightTheme.SECONDARY)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create a grid of buttons (2x2)
        for i, (text, cmd) in enumerate(actions):
            row, col = divmod(i, 2)
            btn = tk.Button(buttons_frame, text=text, command=cmd,
                            fg=LightTheme.BUTTON_TEXT, **self.create_button_style())
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

        # Make columns equally sized
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)

        # Info section
        info_frame = tk.LabelFrame(self.sidebar_frame, text="Information",
                                   font=("Segoe UI", 11),
                                   fg=LightTheme.TEXT_SECONDARY,
                                   bg=LightTheme.SECONDARY,
                                   bd=1, highlightbackground=LightTheme.BORDER_COLOR)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 15))

        # Info text
        info_text = "Drag nodes to position them interactively.\n\n" + \
                    "Format examples:\n" + \
                    "Places: p1, p2, p3\n" + \
                    "Transitions: t1, t2\n" + \
                    "Arcs: p1->t1, t1->p2\n" + \
                    "Markings: p1=1, p2=0"

        info_label = tk.Label(info_frame, text=info_text,
                              justify=tk.LEFT, anchor='w',
                              fg=LightTheme.TEXT_SECONDARY,
                              bg=LightTheme.SECONDARY,
                              font=("Segoe UI", 9))
        info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def toggle_sidebar(self):
        if self.sidebar_visible:
            # Hide sidebar
            self.sidebar_frame.pack_forget()
            self.toggle_btn.config(text="≡ Show Panel")
        else:
            # Show sidebar
            self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, before=self.visualization_container)
            self.toggle_btn.config(text="≡ Hide Panel")

        self.sidebar_visible = not self.sidebar_visible
        # Redraw canvas to adjust to new size
        self.canvas.draw()

    def create_button_style(self):
        return {
            **LightTheme.BUTTON_STYLE,
            "bg": LightTheme.BUTTON_BG,
            "activebackground": LightTheme.BUTTON_HOVER,
            "activeforeground": LightTheme.BUTTON_TEXT
        }

    def setup_graph(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(LightTheme.GRAPH_BG)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._draw_network()

    def _draw_network(self):
        self.ax.clear()
        G = self.petri_net.get_networkx_graph()

        pos = nx.get_node_attributes(G, 'pos')

        # Draw edges
        for edge in G.edges():
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=self.ax,
                                   edge_color=LightTheme.EDGE_COLOR, width=1.5,
                                   arrowsize=15, arrowstyle='->')

        # Draw nodes
        place_nodes = [node for node in G.nodes() if node in self.petri_net.places]
        transition_nodes = [node for node in G.nodes() if node in self.petri_net.transitions]

        # Draw places (circular)
        nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, ax=self.ax,
                               node_color=LightTheme.PLACE_COLOR,
                               node_size=1000,
                               edgecolors="#ffffff", linewidths=1.5)

        # Draw transitions (square)
        for node in transition_nodes:
            x, y = pos[node]
            square_size = 0.14
            square = plt.Rectangle((x - square_size / 2, y - square_size / 2),
                                   square_size, square_size,
                                   facecolor=LightTheme.TRANSITION_COLOR,
                                   edgecolor="#ffffff",
                                   linewidth=1.5, zorder=2)
            self.ax.add_patch(square)

        # Draw token counts inside places
        for node in place_nodes:
            x, y = pos[node]
            tokens = self.petri_net.places[node].tokens
            if tokens > 0:
                self.ax.text(x, y, str(tokens),
                             fontsize=10, color="#ffffff",
                             ha='center', va='center', fontweight='bold')

        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=self.ax, font_color=LightTheme.TEXT_COLOR,
                                font_size=9, font_family="Segoe UI", font_weight='bold')

        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)

        # Add light grid
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color=LightTheme.EDGE_COLOR)

        self.canvas.draw()

    def generate_network(self):
        try:
            self.petri_net = PetriNet()

            # Parse markings
            marking_input = self.marking_entry.get().strip()
            markings = {}
            if marking_input:
                markings = dict(m.split('=') for m in marking_input.split(','))

            # Add places
            places_input = self.places_entry.get().strip()
            if places_input:
                for place in places_input.split(','):
                    name = place.strip()
                    tokens = int(markings.get(name, 0))
                    self.petri_net.add_place(name, tokens)

            # Add transitions
            transitions_input = self.transitions_entry.get().strip()
            if transitions_input:
                for transition in transitions_input.split(','):
                    self.petri_net.add_transition(transition.strip())

            # Add arcs
            arcs_input = self.arcs_entry.get().strip()
            if arcs_input:
                for arc in arcs_input.split(','):
                    src, dst = arc.strip().split('->')
                    self.petri_net.add_arc(src.strip(), dst.strip())

            self._draw_network()
            self._setup_dragging()

            # Modern success notification
            self.show_notification("Network Generated", "Network was successfully created.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input format: {str(e)}", parent=self.root)

    def _setup_dragging(self):
        def on_press(event):
            if event.inaxes != self.ax: return
            for node in self.petri_net.pos:
                x, y = self.petri_net.pos[node]
                if np.hypot(event.xdata - x, event.ydata - y) < 0.1:
                    self.dragging = node
                    self.root.config(cursor="fleur")  # Change cursor to indicate dragging
                    break

        def on_motion(event):
            if not self.dragging or event.inaxes != self.ax: return
            self.petri_net.pos[self.dragging] = (event.xdata, event.ydata)
            self._draw_network()

        def on_release(event):
            self.dragging = None
            self.root.config(cursor="")  # Reset cursor

        self.canvas.mpl_connect('button_press_event', on_press)
        self.canvas.mpl_connect('motion_notify_event', on_motion)
        self.canvas.mpl_connect('button_release_event', on_release)

    def show_notification(self, title, message):
        """Show a temporary notification that fades away"""
        notification = tk.Toplevel(self.root)
        notification.overrideredirect(True)

        # Position in the center of the visualization area
        x = self.root.winfo_x() + self.visualization_container.winfo_x() + (
                    self.visualization_container.winfo_width() // 2) - 150
        y = self.root.winfo_y() + self.visualization_container.winfo_y() + 50
        notification.geometry(f"300x80+{x}+{y}")

        # Create a frame with light blue background and rounded edges
        frame = tk.Frame(notification, bg=LightTheme.ACCENT_LIGHT, padx=15, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Add title and message
        tk.Label(frame, text=title, font=("Segoe UI", 12, "bold"),
                 fg="white", bg=LightTheme.ACCENT_LIGHT).pack(anchor="w")
        tk.Label(frame, text=message, font=("Segoe UI", 10),
                 fg="white", bg=LightTheme.ACCENT_LIGHT).pack(anchor="w", pady=(5, 0))

        # Auto-close after 2 seconds
        notification.after(2000, notification.destroy)

    def load_net_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Petri Net Files", "*.net"), ("All Files", "*.*")])
        if not file_path: return

        self.petri_net = PetriNet()
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    if parts[0] == 'PLACE':
                        name = parts[1]
                        tokens = int(parts[3]) if len(parts) > 3 else 0
                        self.petri_net.add_place(name, tokens)
                    elif parts[0] == 'TRANSITION':
                        self.petri_net.add_transition(parts[1])
                    elif parts[0] == 'ARC':
                        src, dst = parts[1], parts[2]
                        weight = int(parts[3]) if len(parts) > 3 else 1
                        self.petri_net.add_arc(src, dst, weight)

            self._update_form_fields()
            self._draw_network()
            self._setup_dragging()
            self.show_notification("File Loaded", f"Successfully loaded model")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}", parent=self.root)

    def _update_form_fields(self):
        self.places_entry.delete(0, tk.END)
        self.places_entry.insert(0, ", ".join(self.petri_net.places.keys()))

        self.transitions_entry.delete(0, tk.END)
        self.transitions_entry.insert(0, ", ".join(self.petri_net.transitions.keys()))

        self.arcs_entry.delete(0, tk.END)
        self.arcs_entry.insert(0, ", ".join(f"{src}->{dst}" for src, dst, _ in self.petri_net.arcs))

        self.marking_entry.delete(0, tk.END)
        self.marking_entry.insert(0, ", ".join(f"{name}={p.tokens}"
                                               for name, p in self.petri_net.places.items()))

    def analyze_net(self):
        results_text = "Network Analysis:\n\n"

        # Basic properties
        results_text += f"Bounded: {'Yes' if self.petri_net.is_bounded() else 'No'}\n"
        results_text += f"Live Transitions: {'Yes' if self.petri_net.has_live_transitions() else 'No'}\n\n"

        # Places and their tokens
        results_text += "Places Status:\n"
        for name, p in self.petri_net.places.items():
            results_text += f"  {name}: {p.tokens} tokens\n"

        # Display in a more modern dialog with light theme
        top = tk.Toplevel(self.root)
        top.title("Analysis Results")
        top.geometry("400x300")
        top.configure(bg=LightTheme.PRIMARY)
        top.transient(self.root)
        top.grab_set()

        # Results display
        frame = tk.Frame(top, bg=LightTheme.PRIMARY, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Header with gradient line underneath
        header_frame = tk.Frame(frame, bg=LightTheme.PRIMARY)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(header_frame, text="Analysis Results", font=("Segoe UI", 16, "bold"),
                 fg=LightTheme.ACCENT_DARK, bg=LightTheme.PRIMARY).pack(anchor="w")

        # Gradient line
        gradient_line = tk.Canvas(header_frame, height=2, bg=LightTheme.PRIMARY,
                                  highlightthickness=0)
        gradient_line.pack(fill=tk.X, pady=(5, 0))
        gradient_line.create_rectangle(0, 0, 400, 2,
                                       fill=LightTheme.ACCENT_MEDIUM, outline="")

        # Results text
        results = tk.Text(frame, wrap=tk.WORD, height=10,
                          bg=LightTheme.ENTRY_BG, fg=LightTheme.TEXT_COLOR,
                          relief=tk.SOLID, bd=1, padx=10, pady=10)
        results.pack(fill=tk.BOTH, expand=True)
        results.insert(tk.END, results_text)
        results.config(state=tk.DISABLED)

        # Close button
        btn = tk.Button(frame, text="Close", command=top.destroy,
                        fg=LightTheme.BUTTON_TEXT, **self.create_button_style())
        btn.pack(pady=(15, 0))

    def reset_model(self):
        result = messagebox.askyesno("Confirm Reset",
                                     "This will delete all data. Are you sure?",
                                     parent=self.root)
        if not result:
            return

        self.petri_net = PetriNet()
        for entry in [self.places_entry, self.transitions_entry,
                      self.arcs_entry, self.marking_entry]:
            entry.delete(0, tk.END)
        self._draw_network()
        self.show_notification("Reset Complete", "All data has been cleared")


if __name__ == "__main__":
    root = tk.Tk()
    app = PetriNetApp(root)
    root.mainloop()
