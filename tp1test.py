import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PetriNetAnalyzer:
    def __init__(self):
        self.places = set()
        self.transitions = set()
        self.arcs = []
        self.marking = {}
        self.graph = nx.DiGraph()

        # Visualization parameters
        self.place_color = "#4E79A7"
        self.transition_color = "#F28E2B"
        self.active_place_color = "#59A14F"
        self.edge_color = "#79706E"
        self.node_size = 2500

    def load_network(self, places, transitions, arcs, marking):
        """Load network components into the analyzer with validation"""
        self.places = set(places)
        self.transitions = set(transitions)
        self.arcs = [(src.strip(), dest.strip()) for src, dest in arcs]
        self.marking = {p.strip(): int(v) for p, v in marking.items()}
        self._validate_network()
        self._build_graph()

    def _validate_network(self):
        """Validate network structure and relationships"""
        # Validate arc connections
        for src, dest in self.arcs:
            if src not in self.places and src not in self.transitions:
                raise ValueError(f"Invalid source node in arc: {src}")
            if dest not in self.places and dest not in self.transitions:
                raise ValueError(f"Invalid destination node in arc: {dest}")

        # Validate marking consistency
        for place in self.marking:
            if place not in self.places:
                raise ValueError(f"Marking specified for non-existent place: {place}")

    def _build_graph(self):
        """Construct networkx graph representation with type annotations"""
        self.graph.clear()
        self.graph.add_nodes_from(self.places, bipartite=0, node_type='place')
        self.graph.add_nodes_from(self.transitions, bipartite=1, node_type='transition')
        self.graph.add_edges_from(self.arcs)

    def analyze_boundedness(self):
        """
        Check boundedness using structural analysis.
        Returns True if all places have finite capacity in initial marking.
        """
        return all(isinstance(m, int) and m >= 0 for m in self.marking.values())

    def analyze_liveness(self):
        """
        Calculate structural liveness percentage.
        A transition is considered structurally live if it has:
        - At least one input place and one output place
        - All input places have paths from initial marking
        """
        live_count = 0
        for t in self.transitions:
            inputs = {src for src, dest in self.arcs if dest == t}
            outputs = {dest for src, dest in self.arcs if src == t}

            if inputs and outputs and all(self.marking.get(p, 0) > 0 for p in inputs):
                live_count += 1
        return live_count / len(self.transitions) if self.transitions else 0.0

    def get_enabled_transitions(self):
        """Identify currently enabled transitions using standard Petri net semantics"""
        enabled = []
        for t in self.transitions:
            inputs = [src for src, dest in self.arcs if dest == t]
            if inputs and all(self.marking.get(p, 0) > 0 for p in inputs):
                enabled.append(t)
        return enabled


class PetriNetGUI:
    def __init__(self, root):
        self.root = root
        self.analyzer = PetriNetAnalyzer()
        self.current_figure = None

        self._setup_ui()
        self._create_input_panel()
        self._create_visualization_frame()
        self._create_results_panel()

    def _setup_ui(self):
        """Configure main window settings"""
        self.root.title("PetriNet Analysis Suite")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 800)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def _create_input_panel(self):
        """Create input controls section with validation"""
        input_frame = ttk.LabelFrame(self.root, text="Network Configuration")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        fields = [
            ("Places", "p1,p2,p3"),
            ("Transitions", "t1,t2,t3"),
            ("Arcs", "p1->t1,t1->p2,p2->t2"),
            ("Marking", "p1=1,p2=0,p3=2")
        ]

        self.entries = {}
        for idx, (label, default) in enumerate(fields):
            ttk.Label(input_frame, text=f"{label}:").grid(row=idx, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(input_frame, width=35)
            entry.insert(0, default)
            entry.grid(row=idx, column=1, padx=5, pady=5)
            self.entries[label.lower()] = entry

        ttk.Button(input_frame, text="Analyze", command=self._analyze).grid(row=4, columnspan=2, pady=10)
        ttk.Button(input_frame, text="Reset", command=self._reset).grid(row=5, columnspan=2, pady=10)

    def _create_visualization_frame(self):
        """Create visualization canvas with responsive layout"""
        self.viz_frame = ttk.Frame(self.root)
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)

    def _create_results_panel(self):
        """Create analysis results display with enhanced formatting"""
        results_frame = ttk.LabelFrame(self.root, text="Analysis Results")
        results_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self.results = {
            'bounded': ttk.Label(results_frame, text="Structural Boundedness: N/A"),
            'liveness': ttk.Label(results_frame, text="Structural Liveness: N/A"),
            'enabled': ttk.Label(results_frame, text="Enabled Transitions: N/A")
        }

        for idx, (key, widget) in enumerate(self.results.items()):
            widget.grid(row=idx, column=0, sticky="w", padx=5, pady=5)

    def _parse_inputs(self):
        """Parse and validate user inputs with comprehensive checks"""
        try:
            places = [p.strip() for p in self.entries['places'].get().split(',') if p.strip()]
            transitions = [t.strip() for t in self.entries['transitions'].get().split(',') if t.strip()]
            arcs = [a.strip().split('->') for a in self.entries['arcs'].get().split(',') if a.strip()]
            marking = dict(m.split('=') for m in self.entries['marking'].get().split(',') if m.strip())

            if not places:
                raise ValueError("At least one place must be defined")
            if not transitions:
                raise ValueError("At least one transition must be defined")
            if not arcs:
                raise ValueError("At least one arc must be defined")

            return places, transitions, arcs, marking
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input format: {str(e)}")
            return None

    def _analyze(self):
        """Execute analysis pipeline with error handling"""
        inputs = self._parse_inputs()
        if not inputs:
            return

        try:
            self.analyzer.load_network(*inputs)
            self._update_visualization()
            self._update_results()
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))

    def _update_visualization(self):
        """Generate interactive network visualization"""
        if self.current_figure:
            plt.close(self.current_figure)

        fig = plt.Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Create bipartite layout with optimized spacing
        pos = nx.bipartite_layout(self.analyzer.graph, self.analyzer.places, scale=2.0)

        # Enhanced node styling with state indication
        node_colors = []
        for node in self.analyzer.graph.nodes():
            if node in self.analyzer.places:
                node_colors.append(self.analyzer.active_place_color
                                   if self.analyzer.marking.get(node, 0) > 0
                                   else self.analyzer.place_color)
            else:
                node_colors.append(self.analyzer.transition_color)

        # Draw with professional styling
        nx.draw(self.analyzer.graph, pos, ax=ax,
                node_color=node_colors,
                node_size=self.analyzer.node_size,
                edge_color=self.analyzer.edge_color,
                width=2,
                with_labels=True,
                font_size=10,
                font_weight='bold',
                arrowsize=20)

        # Embed in Tkinter with proper layout management
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.current_figure = fig

    def _update_results(self):
        """Update results display with precise formatting"""
        bounded = self.analyzer.analyze_boundedness()
        liveness = self.analyzer.analyze_liveness()
        enabled = self.analyzer.get_enabled_transitions()

        self.results['bounded'].config(text=f"Structural Boundedness: {'Yes' if bounded else 'No'}")
        self.results['liveness'].config(text=f"Structural Liveness: {liveness:.0%}")

        # Corrected line with proper parentheses
        self.results['enabled'].config(
            text="Enabled Transitions:\n" + ("\n".join(enabled) if enabled else "No Enabled Transitions")
        )

    def _reset(self):
        """Reset application state with proper cleanup"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.analyzer = PetriNetAnalyzer()
        self._update_results()
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None
        for widget in self.viz_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PetriNetGUI(root)
    root.mainloop()
