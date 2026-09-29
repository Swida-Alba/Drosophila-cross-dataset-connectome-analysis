import os
import json
import numpy as np

def generate_interactive_heatmap(matrices_dict, filename, title='', showfig=True, fontsize=12, verbose=True):
    """
    Create interactive heatmap for comparison metrics.
    
    Parameters
    ----------
    matrices_dict : dict
        Dictionary where keys are metric names (e.g., 'jaccard', 'cosine') and values are pandas DataFrames.
    filename : str
        Output HTML filename.
    title : str
        Title for the heatmap.
    showfig : bool
        Whether to open in browser.
    fontsize : int
        Default font size.
    verbose : bool
        Whether to print progress messages.
    """
    
    available_metrics = list(matrices_dict.keys())
    if not available_metrics:
        raise ValueError("matrices_dict cannot be empty")
        
    default_metric = available_metrics[0]
    
    # Prepare data for JS
    matrices_data = {}
    for metric, df in matrices_dict.items():
        # Round to 4 decimal places for similarity metrics
        # Handle NaN values by replacing with 0
        values = df.values
        values = np.nan_to_num(values, nan=0.0)
        matrices_data[metric] = np.round(values, 4).tolist()
        
    # Use the first matrix to get labels and shape
    first_df = matrices_dict[default_metric]
    x_labels = first_df.columns.astype(str).tolist()
    y_labels = first_df.index.astype(str).tolist()
    
    is_large = first_df.shape[0] > 100 or first_df.shape[1] > 100
    
    # Clustering logic
    if verbose:
        print("  Computing hierarchical clustering...")
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    
    clustering_results = {}
    clustering_successful = False
    row_order_clustered = list(range(len(y_labels)))
    col_order_clustered = list(range(len(x_labels)))
    
    try:
        # Use the default metric for clustering
        data_for_clustering = np.nan_to_num(first_df.values, nan=0.0)
        
        # Cluster rows
        if data_for_clustering.shape[0] > 1:
            row_distances = pdist(data_for_clustering, metric='euclidean')
            row_linkage = linkage(row_distances, method='ward')
            row_order_clustered = leaves_list(row_linkage).tolist()
        else:
            row_order_clustered = [0]
            
        # Cluster columns
        if data_for_clustering.shape[1] > 1:
            col_distances = pdist(data_for_clustering.T, metric='euclidean')
            col_linkage = linkage(col_distances, method='ward')
            col_order_clustered = leaves_list(col_linkage).tolist()
        else:
            col_order_clustered = [0]
            
        clustering_results['ward'] = {
            'row_order': row_order_clustered,
            'col_order': col_order_clustered
        }
        clustering_successful = True
        print(f"  ✓ Clustering complete")
    except Exception as e:
        print(f"  ⚠ Clustering failed: {e}")

    row_order_original = list(range(len(y_labels)))
    col_order_original = list(range(len(x_labels)))
    
    # Generate unique storage key
    from datetime import datetime
    output_name = os.path.splitext(os.path.basename(filename))[0]
    timestamp_hash = datetime.now().strftime('%Y%m%d%H%M%S')
    storage_key = f"heatmap_settings_{output_name}#{timestamp_hash}"
    
    # Metric display names
    metric_display_names = {m: m.replace('_', ' ').title() for m in available_metrics}
    
    # Generate HTML options for metric select
    metric_options = ""
    for m in available_metrics:
        selected = " selected" if m == default_metric else ""
        metric_options += f'<option value="{m}"{selected}>{metric_display_names[m]}</option>'

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .main-container {{ max-width: 1800px; margin: 0 auto; }}
        .controls {{ background: white; padding: 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 15px; }}
        .controls-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-bottom: 10px; }}
        .control-section {{ background: #f8f9fa; padding: 8px; border-radius: 4px; border: 1px solid #e9ecef; }}
        .control-section h3 {{ margin: 0 0 8px 0; font-size: 12px; font-weight: 600; color: #495057; text-transform: uppercase; }}
        .button-group {{ display: flex; gap: 4px; flex-wrap: wrap; }}
        button {{ padding: 6px 10px; border: 1px solid #dee2e6; background: white; border-radius: 3px; cursor: pointer; font-size: 11px; }}
        button:hover {{ background: #f8f9fa; }}
        button.active {{ background: #4CAF50; color: white; border-color: #4CAF50; }}
        select {{ width: 100%; padding: 4px; border: 1px solid #dee2e6; border-radius: 3px; font-size: 11px; }}
        #heatmap-container {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        #heatmap {{ width: 100%; height: 800px; }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="controls">
            <div class="controls-grid">
                <div class="control-section">
                    <h3>📊 Metric & Ordering</h3>
                    <div style="margin-bottom: 8px;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Metric:</label>
                        <select id="metricSelect" onchange="updateMetric()">
                            {metric_options}
                        </select>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Ordering:</label>
                        <div class="button-group">
                            <button id="btn-original" class="active" onclick="toggleClustering('original')">Original</button>
                            <button id="btn-clustered" onclick="toggleClustering('clustered')">Clustered</button>
                        </div>
                    </div>
                </div>
                
                <div class="control-section">
                    <h3>🎨 Color</h3>
                    <select id="colorscaleSelect" onchange="updateColorscale()" style="margin-bottom: 8px;">
                        <option value="Viridis">Viridis</option>
                        <option value="Plasma">Plasma</option>
                        <option value="Inferno">Inferno</option>
                        <option value="Magma">Magma</option>
                        <option value="Greens">Greens</option>
                        <option value="Blues">Blues</option>
                        <option value="Reds">Reds</option>
                        <option value="RdBu">Red-Blue (Diverging)</option>
                    </select>
                </div>

                <div class="control-section">
                    <h3>🎚️ Display</h3>
                    <div style="margin-bottom: 6px;">
                        <label style="font-size: 10px;">Font Size: <span id="fontSizeValue">{fontsize}px</span></label>
                        <input type="range" min="8" max="24" value="{fontsize}" oninput="updateFontSize(this.value)" style="width: 100%;">
                    </div>
                    <button onclick="toggleLabels()" id="toggleLabelsBtn" style="width: 100%;">{'🏷️ Show Labels' if is_large else '🏷️ Hide Labels'}</button>
                </div>
            </div>
        </div>
        
        <div id="heatmap-container">
            <div id="heatmap"></div>
        </div>
    </div>

    <script>
        const metricsData = {json.dumps(matrices_data)};
        const metricDisplayNames = {json.dumps(metric_display_names)};
        const xLabels = {json.dumps(x_labels)};
        const yLabels = {json.dumps(y_labels)};
        
        const rowOrderOriginal = {json.dumps(row_order_original)};
        const colOrderOriginal = {json.dumps(col_order_original)};
        const rowOrderClustered = {json.dumps(row_order_clustered)};
        const colOrderClustered = {json.dumps(col_order_clustered)};
        const clusteringAvailable = {json.dumps(clustering_successful)};
        
        let currentMetric = '{default_metric}';
        let currentColorscale = 'Viridis';
        let currentFontSize = {fontsize};
        let showLabels = {json.dumps(not is_large)};
        let useClusteredOrder = false;
        
        function createHeatmap() {{
            const data = metricsData[currentMetric];
            
            // Apply ordering
            let plotData = [];
            let plotX = [];
            let plotY = [];
            
            const rowOrder = useClusteredOrder ? rowOrderClustered : rowOrderOriginal;
            const colOrder = useClusteredOrder ? colOrderClustered : colOrderOriginal;
            
            for (let i = 0; i < rowOrder.length; i++) {{
                let row = [];
                for (let j = 0; j < colOrder.length; j++) {{
                    row.push(data[rowOrder[i]][colOrder[j]]);
                }}
                plotData.push(row);
                plotY.push(yLabels[rowOrder[i]]);
            }}
            
            for (let j = 0; j < colOrder.length; j++) {{
                plotX.push(xLabels[colOrder[j]]);
            }}
            
            const trace = {{
                z: plotData,
                x: plotX,
                y: plotY,
                type: 'heatmap',
                colorscale: currentColorscale,
                colorbar: {{
                    title: metricDisplayNames[currentMetric],
                    titleside: 'right'
                }},
                hovertemplate: '<b>Source:</b> %{{y}}<br><b>Target:</b> %{{x}}<br><b>Value:</b> %{{z:.4f}}<extra></extra>'
            }};
            
            const layout = {{
                title: '{title}',
                font: {{ size: currentFontSize }},
                xaxis: {{
                    title: 'Target',
                    showticklabels: showLabels,
                    tickangle: -45
                }},
                yaxis: {{
                    title: 'Source',
                    showticklabels: showLabels,
                    autorange: 'reversed',
                    // Square cells: lock the cell aspect to 1:1 whatever the
                    // container does (same contract as report_kit's
                    // square_cells figures).
                    scaleanchor: 'x',
                    scaleratio: 1,
                    constrain: 'domain'
                }},
                margin: {{ l: 150, b: 100 }}
            }};
            
            Plotly.newPlot('heatmap', [trace], layout);
        }}
        
        function updateMetric() {{
            currentMetric = document.getElementById('metricSelect').value;
            createHeatmap();
        }}
        
        function updateColorscale() {{
            currentColorscale = document.getElementById('colorscaleSelect').value;
            createHeatmap();
        }}
        
        function updateFontSize(val) {{
            currentFontSize = parseInt(val);
            document.getElementById('fontSizeValue').textContent = val + 'px';
            createHeatmap();
        }}
        
        function toggleLabels() {{
            showLabels = !showLabels;
            document.getElementById('toggleLabelsBtn').textContent = showLabels ? '🏷️ Hide Labels' : '🏷️ Show Labels';
            createHeatmap();
        }}
        
        function toggleClustering(mode) {{
            useClusteredOrder = (mode === 'clustered');
            document.getElementById('btn-original').classList.toggle('active', mode === 'original');
            document.getElementById('btn-clustered').classList.toggle('active', mode === 'clustered');
            
            if (useClusteredOrder && !clusteringAvailable) {{
                alert('Clustering not available');
                toggleClustering('original');
                return;
            }}
            createHeatmap();
        }}
        
        // Initialize
        createHeatmap();
    </script>
</body>
</html>
'''
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    if showfig:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(filename))
