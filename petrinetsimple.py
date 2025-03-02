import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


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


class MinimalisticTheme:
    # Main colors
    BG = "#ffffff"
    FG = "#333333"
    
    # Accent colors
    ACCENT = "#4a6fa5"
    LIGHT_ACCENT = "#e8f0fe"
    
    # Node colors
    PLACE_COLOR = "#4a6fa5"
    TRANSITION_COLOR = "#6c757d"
    
    # Graph colors
    GRAPH_BG = "#f8f9fa"
    EDGE_COLOR = "#adb5bd"


class PetriNetApp:
    def __init__(self, root):
        self.root = root
        self.petri_net = PetriNet()
        self.dragging = None
        self.setup_ui()
        self.setup_graph()

    def setup_ui(self):
        self.root.title("PetriNet")
        self.root.geometry("1200x700")
        self.root.configure(bg=MinimalisticTheme.BG)

        # Create main frame with horizontal layout
        self.main_frame = tk.Frame(self.root, bg=MinimalisticTheme.BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for inputs
        self.input_frame = tk.Frame(self.main_frame, bg=MinimalisticTheme.BG, width=250)
        self.input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.input_frame.pack_propagate(False)
        
        # Right panel for visualization
        self.viz_frame = tk.Frame(self.main_frame, bg=MinimalisticTheme.BG)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup input panel
        self.setup_input_panel()
        
        # Create canvas for network visualization
        self.figure = plt.figure(figsize=(8, 6), facecolor=MinimalisticTheme.GRAPH_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_input_panel(self):
        # Title
        tk.Label(self.input_frame, text="PetriNet", 
                font=("Arial", 16, "bold"),
                fg=MinimalisticTheme.ACCENT, 
                bg=MinimalisticTheme.BG).pack(pady=(0, 20))
        
        # Input fields
        input_fields = [
            ("Places", 'places_entry'),
            ("Transitions", 'transitions_entry'),
            ("Arcs", 'arcs_entry'),
            ("Markings", 'marking_entry')
        ]
        
        for text, var_name in input_fields:
            frame = tk.Frame(self.input_frame, bg=MinimalisticTheme.BG)
            frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(frame, text=text, font=("Arial", 10),
                    fg=MinimalisticTheme.FG, bg=MinimalisticTheme.BG,
                    anchor='w').pack(fill=tk.X)
            
            entry = tk.Entry(frame, bg=MinimalisticTheme.LIGHT_ACCENT, 
                            fg=MinimalisticTheme.FG,
                            relief=tk.FLAT, bd=0)
            entry.pack(fill=tk.X, ipady=5, pady=(3, 0))
            setattr(self, var_name, entry)
        
        # Buttons frame
        btn_frame = tk.Frame(self.input_frame, bg=MinimalisticTheme.BG)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Action buttons
        tk.Button(btn_frame, text="Generate", command=self.generate_network,
                bg=MinimalisticTheme.ACCENT, fg="white",
                relief=tk.FLAT, bd=0, padx=0, pady=5,
                font=("Arial", 10)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
                
        tk.Button(btn_frame, text="Analyze", command=self.analyze_net,
                bg=MinimalisticTheme.ACCENT, fg="white",
                relief=tk.FLAT, bd=0, padx=0, pady=5,
                font=("Arial", 10)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
                
        tk.Button(btn_frame, text="Reset", command=self.reset_model,
                bg=MinimalisticTheme.ACCENT, fg="white",
                relief=tk.FLAT, bd=0, padx=0, pady=5,
                font=("Arial", 10)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Help text
        help_frame = tk.Frame(self.input_frame, bg=MinimalisticTheme.BG)
        help_frame.pack(fill=tk.X, pady=(20, 0))
        
        help_text = "Format examples:\n" + \
                   "Places: p1, p2, p3\n" + \
                   "Transitions: t1, t2\n" + \
                   "Arcs: p1->t1, t1->p2\n" + \
                   "Markings: p1=1, p2=0"
                   
        tk.Label(help_frame, text=help_text,
                justify=tk.LEFT, anchor='w',
                fg=MinimalisticTheme.FG,
                bg=MinimalisticTheme.BG,
                font=("Arial", 8)).pack(fill=tk.X)

    def setup_graph(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(MinimalisticTheme.GRAPH_BG)
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
                                  edge_color=MinimalisticTheme.EDGE_COLOR, width=1.5,
                                  arrowsize=15, arrowstyle='->')
        
        # Draw nodes
        place_nodes = [node for node in G.nodes() if node in self.petri_net.places]
        transition_nodes = [node for node in G.nodes() if node in self.petri_net.transitions]
        
        # Draw places (circular)
        nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, ax=self.ax,
                              node_color=MinimalisticTheme.PLACE_COLOR,
                              node_size=800,
                              edgecolors="#ffffff", linewidths=1)
        
        # Draw transitions (square)
        for node in transition_nodes:
            x, y = pos[node]
            square_size = 0.12
            square = plt.Rectangle((x - square_size / 2, y - square_size / 2),
                                  square_size, square_size,
                                  facecolor=MinimalisticTheme.TRANSITION_COLOR,
                                  edgecolor="#ffffff",
                                  linewidth=1, zorder=2)
            self.ax.add_patch(square)
        
        # Draw token counts inside places
        for node in place_nodes:
            x, y = pos[node]
            tokens = self.petri_net.places[node].tokens
            if tokens > 0:
                self.ax.text(x, y, str(tokens),
                            fontsize=9, color="#ffffff",
                            ha='center', va='center', fontweight='bold')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=self.ax, font_color=MinimalisticTheme.FG,
                               font_size=9, font_family="Arial")
        
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        
        self.canvas.draw()
        self._setup_dragging()

    def _setup_dragging(self):
        def on_press(event):
            if event.inaxes != self.ax: return
            for node in self.petri_net.pos:
                x, y = self.petri_net.pos[node]
                if np.hypot(event.xdata - x, event.ydata - y) < 0.1:
                    self.dragging = node
                    self.root.config(cursor="fleur")
                    break
                    
        def on_motion(event):
            if not self.dragging or event.inaxes != self.ax: return
            self.petri_net.pos[self.dragging] = (event.xdata, event.ydata)
            self._draw_network()
            
        def on_release(event):
            self.dragging = None
            self.root.config(cursor="")
            
        self.canvas.mpl_connect('button_press_event', on_press)
        self.canvas.mpl_connect('motion_notify_event', on_motion)
        self.canvas.mpl_connect('button_release_event', on_release)

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
                    tokens = int(markings.get(name, 0)) if name in markings else 0
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
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def analyze_net(self):
        if not self.petri_net.places:
            messagebox.showinfo("Info", "No network to analyze")
            return
            
        results = f"Bounded: {'Yes' if self.petri_net.is_bounded() else 'No'}\n"
        results += f"Live Transitions: {'Yes' if self.petri_net.has_live_transitions() else 'No'}\n\n"
        
        results += "Places Status:\n"
        for name, p in self.petri_net.places.items():
            results += f"  {name}: {p.tokens} tokens\n"
            
        # Simple analysis dialog
        top = tk.Toplevel(self.root)
        top.title("Analysis")
        top.geometry("300x250")
        top.configure(bg=MinimalisticTheme.BG)
        top.transient(self.root)
        top.grab_set()
        
        # Results display
        tk.Label(top, text="Analysis Results", 
                font=("Arial", 12, "bold"),
                fg=MinimalisticTheme.ACCENT, 
                bg=MinimalisticTheme.BG).pack(pady=(15, 10))
                
        results_text = tk.Text(top, wrap=tk.WORD, height=10,
                               bg=MinimalisticTheme.LIGHT_ACCENT, 
                               fg=MinimalisticTheme.FG,
                               relief=tk.FLAT, bd=0, padx=10, pady=10)
        results_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        results_text.insert(tk.END, results)
        results_text.config(state=tk.DISABLED)
        
        # Close button
        tk.Button(top, text="Close", command=top.destroy,
                  bg=MinimalisticTheme.ACCENT, fg="white",
                  relief=tk.FLAT, bd=0, padx=20, pady=5).pack(pady=(0, 15))

    def reset_model(self):
        self.petri_net = PetriNet()
        for entry in [self.places_entry, self.transitions_entry, 
                     self.arcs_entry, self.marking_entry]:
            entry.delete(0, tk.END)
        self._draw_network()


if __name__ == "__main__":
    root = tk.Tk()
    app = PetriNetApp(root)
    root.mainloop()
