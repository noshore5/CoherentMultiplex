// Sigma.js graph of nodes A-H
// Use global Sigma and graphology loaded via <script>
if (!window.Sigma || !window.graphology) {
    console.error('Sigma.js or graphology not loaded!');
} else {
    const Sigma = window.Sigma;
    const Graph = window.graphology;
    // Create the graphology graph
    const graph = new Graph();
    // Use hex colors for each node to ensure browser compatibility
    const nodeLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    // Match node colors to signal legend (A: red, B: yellow, C: blue, D: green, E: cyan, F: magenta, G: orange, H: purple)
    // Match node colors to legend: A-red, B-yellow, C-lime, D-green, E-cyan, F-blue, G-purple, H-magenta
    // Match node colors to legend using HSL
    // Match node colors to legend using hex codes for HSL
    const nodeColors = [
        '#ff3333', // A
        '#ff9933', // B
        '#e6ff33', // C
        '#33ff70ff', // D
        '#51fcf0ff', // E
        '#3385ff', // F
        '#9933ff', // G
        '#ff33c6'  // H
    ];
    const N = nodeLabels.length;
    const radius = 1.2; // Reduced for closer spacing
    for (let i = 0; i < N; i++) {
        const angle = (2 * Math.PI * i) / N;
        graph.addNode(nodeLabels[i], {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            size: 30,
            color: nodeColors[i],
            type: 'circle',
            label: nodeLabels[i],
            forceLabel: true,
            labelYOffset: -32, // Move label above node
            labelFont: 'bold 22px Segoe UI, Arial, sans-serif',
            labelColor: '#e0e0e0'
        });
    }
    // Generate unique colormaps for each edge
function getEdgeColor(idx, t) {
    // idx: edge index, t: normalized value [0,1]
    // Use HSL colormap, spread hues evenly for each edge
    const totalEdges = N * (N - 1) / 2;
    // Accent colormap: qualitative, 8 distinct colors
    // Source: https://matplotlib.org/stable/users/prev_whats_new/whats_new_1.5.0.html#accent-colormap
    function accentColormap(t) {
        // 8 colors from matplotlib Accent
        const accentColors = [
            '#e9d1a8ff', // 
            '#b1873fff', // 
            '#777637ff', // 
            '#8aac54ff', // 
            '#378383ff', // 
            '#2e5b80ff', // 
            '#722f83ff', // 
            '#3b1f31ff'  // 
        ];
        // Map t in [0,1] to one of 8 colors
        const idx = Math.floor((1 - t) * accentColors.length);
        return accentColors[Math.max(0, Math.min(accentColors.length - 1, idx))];
    }
    // All edges use the same accent colormap, reversed (color only depends on 1-t)
    return accentColormap(1 - t);
}
    // Initial edge creation
    let edgeIdx = 0;
    for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
            // Initial color: t=0.5
            graph.addEdge(nodeLabels[i], nodeLabels[j], { color: getEdgeColor(edgeIdx, 0.5), size: 4 });
            edgeIdx++;
        }
    }
    // Print edgeDistances to console for debugging whenever it updates
    // Use a polling approach since window.edgeDistances is updated by the HTML script
    let lastDistances = undefined;
    setInterval(() => {
        if (window.edgeDistances !== undefined && window.edgeDistances !== lastDistances) {
            console.log('window.edgeDistances updated:', window.edgeDistances);
            // Update edge widths and colors based on edgeDistances
            let idx = 0;
            for (let i = 0; i < N; i++) {
                for (let j = i + 1; j < N; j++) {
                    let val = window.edgeDistances[idx];
                    // Invert mapping: smaller distance = wider edge
                    // Normalized val: 0 (smallest) to 1 (largest)
                    let width = (typeof val === 'number') ? (12 - val * 10) : 2;
                    let color = getEdgeColor(idx, (typeof val === 'number') ? val : 0.5);
                    if (graph.hasEdge(nodeLabels[i], nodeLabels[j])) {
                        graph.setEdgeAttribute(nodeLabels[i], nodeLabels[j], 'size', width);
                        graph.setEdgeAttribute(nodeLabels[i], nodeLabels[j], 'color', color);
                    }
                    idx++;
                }
            }
            if (typeof sigmaInstance === 'object' && sigmaInstance !== null) {
                sigmaInstance.refresh();
            }
            lastDistances = window.edgeDistances;
        }
    }, 1000);
    // Mount Sigma.js on #sigma-container with custom label renderer
    const container = document.getElementById("sigma-container");
    let sigmaInstance = null;
    if (container) {
        container.style.border = "4px solid #444";
        container.style.borderRadius = "12px";
        // Custom node label renderer
        const customNodeLabelRenderer = function(context, data, settings) {
            if (!data.label) return;
            const font = 'bold 22px Segoe UI, Arial, sans-serif';
            context.font = font;
            context.fillStyle = '#e0e0e0';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            // Position label outside the graph circle
            // Find node index and angle
            const idx = nodeLabels.indexOf(data.label);
            const N = nodeLabels.length;
            const angle = (2 * Math.PI * idx) / N;
            // Place label at a radius outside the node circle
            const graphRadius = 1.2;
            const labelRadius = graphRadius + 0.45; // Move label further out
            const labelX = Math.cos(angle) * labelRadius * 100 + settings.width / 2;
            const labelY = Math.sin(angle) * labelRadius * 100 + settings.height / 2;
            context.fillText(data.label, labelX, labelY);
        };
        sigmaInstance = new Sigma(graph, container, {
            renderLabels: true,
            labelRenderer: customNodeLabelRenderer
        });
        sigmaInstance.getCamera().setState({ ratio: 1.2 });
    }

    // ...existing code...

    // ...existing code...
}
