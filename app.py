import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import re
import zipfile
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Scientific Heatmap Generator",
    page_icon="🔥",
    layout="wide"
)

# ==================== DATA PROCESSING FUNCTIONS ====================
def preprocess_uploaded_content(content: str) -> str:
    """Preprocess uploaded content for handling incomplete cases"""
    lines = content.strip().split('\n')
    processed_lines = []
    last_x_value = None
    
    for line in lines:
        if not line.strip():
            continue
            
        line = line.strip()
        
        if '\t' in line:
            parts = line.split('\t')
        elif ',' in line:
            parts = line.split(',')
        else:
            parts = re.split(r'\s+', line)
        
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 3:
            processed_lines.append(f"{parts[0]},{parts[1]},{parts[2]}")
            last_x_value = parts[0]
        elif len(parts) == 2:
            if last_x_value is not None:
                processed_lines.append(f"{last_x_value},{parts[0]},{parts[1]}")
    
    return '\n'.join(processed_lines)

def parse_data(content: str) -> pd.DataFrame:
    """Parse data from string to DataFrame with robust type handling"""
    processed_content = preprocess_uploaded_content(content)
    
    df = None
    for delimiter in [',', '\t', ';', ' ']:
        try:
            df = pd.read_csv(
                io.StringIO(processed_content), 
                sep=delimiter, 
                header=None,
                engine='python',
                names=['X', 'Y', 'Value']
            )
            if df.shape[1] >= 3:
                df = df.iloc[:, :3]
                df.columns = ['X', 'Y', 'Value']
                break
        except:
            continue
    
    if df is None or df.empty:
        return None
    
    df['X'] = df['X'].astype(str)
    df['Y'] = df['Y'].astype(str)
    
    try:
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    except:
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    df['X_numeric'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y_numeric'] = pd.to_numeric(df['Y'], errors='coerce')
    
    df = df.dropna(subset=['X', 'Y'])
    
    return df

def create_pivot_table(df: pd.DataFrame, sort_numerically: bool = True) -> pd.DataFrame:
    """Create pivot table for heatmap with optional clustering"""
    if df is None or df.empty:
        return None
    
    pivot_df = df.pivot_table(index='Y', columns='X', values='Value', aggfunc='mean')
    
    if sort_numerically:
        try:
            x_values = df[['X', 'X_numeric']].drop_duplicates().dropna(subset=['X_numeric'])
            if not x_values.empty:
                x_sorted = x_values.sort_values('X_numeric')['X'].tolist()
                x_sorted = [x for x in x_sorted if x in pivot_df.columns]
                pivot_df = pivot_df[x_sorted]
        except:
            pass
        
        try:
            y_values = df[['Y', 'Y_numeric']].drop_duplicates().dropna(subset=['Y_numeric'])
            if not y_values.empty:
                y_sorted = y_values.sort_values('Y_numeric')['Y'].tolist()
                y_sorted = [y for y in y_sorted if y in pivot_df.index]
                pivot_df = pivot_df.loc[y_sorted]
        except:
            pass
    
    return pivot_df

def normalize_data(pivot_df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """Normalize data using different methods"""
    if pivot_df is None or pivot_df.empty:
        return None
    
    valid_values = pivot_df.values[~np.isnan(pivot_df.values)]
    if len(valid_values) == 0:
        return pivot_df
    
    if method == 'minmax':
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        if max_val == min_val:
            return pivot_df
        normalized_df = (pivot_df - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        if std_val == 0:
            return pivot_df
        normalized_df = (pivot_df - mean_val) / std_val
    
    elif method == 'log10':
        if np.min(valid_values) <= 0:
            offset = abs(np.min(valid_values)) + 1
            normalized_df = np.log10(pivot_df + offset)
        else:
            normalized_df = np.log10(pivot_df)
    
    elif method == 'percentile':
        normalized_df = pivot_df.rank(axis=1, pct=True)
    
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        normalized_values = scaler.fit_transform(pivot_df.values)
        normalized_df = pd.DataFrame(normalized_values, 
                                   index=pivot_df.index, 
                                   columns=pivot_df.columns)
    
    return normalized_df

# ==================== STATISTICAL FUNCTIONS ====================
def calculate_statistics(pivot_df: pd.DataFrame):
    """Calculate various statistics for rows and columns"""
    stats_dict = {}
    
    # Row statistics
    stats_dict['row_means'] = pivot_df.mean(axis=1, skipna=True)
    stats_dict['row_stds'] = pivot_df.std(axis=1, skipna=True)
    stats_dict['row_medians'] = pivot_df.median(axis=1, skipna=True)
    stats_dict['row_q25'] = pivot_df.quantile(0.25, axis=1)
    stats_dict['row_q75'] = pivot_df.quantile(0.75, axis=1)
    stats_dict['row_min'] = pivot_df.min(axis=1, skipna=True)
    stats_dict['row_max'] = pivot_df.max(axis=1, skipna=True)
    
    # Column statistics
    stats_dict['col_means'] = pivot_df.mean(axis=0, skipna=True)
    stats_dict['col_stds'] = pivot_df.std(axis=0, skipna=True)
    stats_dict['col_medians'] = pivot_df.median(axis=0, skipna=True)
    stats_dict['col_min'] = pivot_df.min(axis=0, skipna=True)
    stats_dict['col_max'] = pivot_df.max(axis=0, skipna=True)
    
    # Global statistics
    stats_dict['global_mean'] = pivot_df.values[~np.isnan(pivot_df.values)].mean()
    stats_dict['global_std'] = pivot_df.values[~np.isnan(pivot_df.values)].std()
    stats_dict['global_min'] = pivot_df.values[~np.isnan(pivot_df.values)].min()
    stats_dict['global_max'] = pivot_df.values[~np.isnan(pivot_df.values)].max()
    
    return stats_dict

def calculate_correlations(pivot_df: pd.DataFrame):
    """Calculate correlation matrix"""
    if pivot_df is None or len(pivot_df) < 2:
        return None
    
    # Handle NaN values
    data_for_corr = pivot_df.fillna(pivot_df.mean())
    
    # Calculate correlation matrix
    corr_matrix = data_for_corr.corr()
    
    return corr_matrix

def perform_statistical_tests(pivot_df: pd.DataFrame, group1_indices=None, group2_indices=None):
    """Perform statistical tests between groups"""
    results = {}
    
    if group1_indices is None or group2_indices is None:
        # Default: compare first half vs second half
        n_rows = len(pivot_df)
        group1_indices = list(range(0, n_rows//2))
        group2_indices = list(range(n_rows//2, n_rows))
    
    # Extract data for groups
    group1_data = pivot_df.iloc[group1_indices].values.flatten()
    group2_data = pivot_df.iloc[group2_indices].values.flatten()
    
    # Remove NaN values
    group1_data = group1_data[~np.isnan(group1_data)]
    group2_data = group2_data[~np.isnan(group2_data)]
    
    if len(group1_data) > 1 and len(group2_data) > 1:
        # t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        results['t_test'] = {'statistic': t_stat, 'p_value': p_value}
        
        # Mann-Whitney U test
        u_stat, p_value_mw = stats.mannwhitneyu(group1_data, group2_data)
        results['mann_whitney'] = {'statistic': u_stat, 'p_value': p_value_mw}
        
        # Effect size (Cohen's d)
        n1, n2 = len(group1_data), len(group2_data)
        pooled_std = np.sqrt(((n1-1)*np.var(group1_data) + (n2-1)*np.var(group2_data)) / (n1+n2-2))
        cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
        results['effect_size'] = {'cohens_d': cohens_d}
    
    return results

# ==================== CLUSTERING FUNCTIONS ====================
def perform_clustering(pivot_df: pd.DataFrame, n_clusters: int = 3, method: str = 'kmeans'):
    """Perform clustering on rows"""
    if pivot_df is None or len(pivot_df) < n_clusters:
        return None
    
    # Handle NaN values
    data_for_clustering = pivot_df.fillna(pivot_df.mean())
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    
    if method == 'kmeans':
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
    elif method == 'hierarchical':
        # Perform hierarchical clustering
        Z = linkage(scaled_data, method='ward')
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    
    return cluster_labels

def perform_biclustering(pivot_df: pd.DataFrame):
    """Simple biclustering visualization"""
    if pivot_df is None:
        return None
    
    # Simple reordering based on hierarchical clustering
    from scipy.cluster.hierarchy import leaves_list
    
    # Row clustering
    row_data = pivot_df.fillna(pivot_df.mean()).values
    row_linkage = linkage(row_data, method='ward')
    row_order = leaves_list(row_linkage)
    
    # Column clustering
    col_data = pivot_df.fillna(pivot_df.mean()).T.values
    col_linkage = linkage(col_data, method='ward')
    col_order = leaves_list(col_linkage)
    
    return row_order, col_order

# ==================== VISUALIZATION FUNCTIONS ====================
def create_basic_heatmap(pivot_df: pd.DataFrame, colorscale: str = 'viridis',
                        x_label: str = "X", y_label: str = "Y",
                        colorbar_title: str = "Value",
                        show_values: bool = False,
                        value_format: str = ".2f"):
    """Create basic heatmap with all original features"""
    
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Prepare data
    plot_data = pivot_df.values.copy()
    
    # Create text for cells if requested
    if show_values:
        text_matrix = np.empty_like(plot_data, dtype=object)
        for i in range(text_matrix.shape[0]):
            for j in range(text_matrix.shape[1]):
                value = plot_data[i, j]
                if np.isnan(value):
                    text_matrix[i, j] = ""
                else:
                    text_matrix[i, j] = format(value, value_format)
    else:
        text_matrix = None
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=plot_data,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale=colorscale,
        text=text_matrix,
        texttemplate='%{text}' if text_matrix is not None else None,
        hoverongaps=False,
        hoverinfo='x+y+z',
        colorbar=dict(
            title=dict(
                text=colorbar_title,
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black', size=10),
            orientation='v',
            x=1.02,
            xpad=20,
            len=0.8
        ),
        xgap=1,
        ygap=1
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Heatmap",
            font=dict(size=16, color='black'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text=x_label,
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12),
            gridcolor='black',
            linecolor='black',
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False,
            type='category',
            tickangle=45
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12),
            gridcolor='black',
            linecolor='black',
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False,
            type='category'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=100, t=50, b=100)
    )
    
    return fig

def create_normalized_heatmap(pivot_df: pd.DataFrame, colorscale: str = 'viridis',
                             x_label: str = "X", y_label: str = "Y",
                             colorbar_title: str = "Value",
                             normalization_method: str = 'minmax'):
    """Create normalized heatmap"""
    
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Normalize data
    normalized_df = normalize_data(pivot_df, normalization_method)
    
    if normalized_df is None:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=normalized_df.values,
        x=normalized_df.columns.tolist(),
        y=normalized_df.index.tolist(),
        colorscale=colorscale,
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text=f"{colorbar_title} ({normalization_method})",
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black', size=10),
            orientation='v',
            x=1.02,
            xpad=20
        ),
        xgap=1,
        ygap=1
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Normalized Heatmap ({normalization_method})",
            font=dict(size=16, color='black'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(color='black', size=14)),
            tickfont=dict(color='black', size=12),
            tickangle=45
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color='black', size=14)),
            tickfont=dict(color='black', size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=100, t=50, b=100)
    )
    
    return fig

def create_smooth_contour(pivot_df: pd.DataFrame, colorscale: str = 'viridis',
                         smoothing_level: float = 1.0,
                         show_contour_lines: bool = True,
                         show_contour_labels: bool = True) -> go.Figure:
    """Create smooth contour plot with all original features"""
    
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Prepare data for contour plot
    x = list(range(len(pivot_df.columns)))
    y = list(range(len(pivot_df.index)))
    z = pivot_df.values
    
    # Handle NaN values
    if np.isnan(z).any():
        z = np.nan_to_num(z)
    
    # Apply smoothing
    if smoothing_level > 0:
        z_smoothed = gaussian_filter(z, sigma=smoothing_level)
    else:
        z_smoothed = z
    
    # Create contour plot
    fig = go.Figure(data=go.Contour(
        z=z_smoothed,
        x=x,
        y=y,
        colorscale=colorscale,
        contours=dict(
            showlabels=show_contour_labels,
            labelfont=dict(size=12, color='black'),
            coloring='heatmap',  # Always filled
            showlines=show_contour_lines,
        ),
        line=dict(width=1 if show_contour_lines else 0),
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text='Value',
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black', size=10),
            orientation='v',
            x=1.02,
            xpad=20
        )
    ))
    
    # Configure axes
    fig.update_xaxes(
        ticktext=pivot_df.columns.tolist(),
        tickvals=x,
        title=dict(
            text='X',
            font=dict(color='black', size=14)
        ),
        tickfont=dict(color='black', size=12),
        gridcolor='black',
        linecolor='black',
        mirror=True,
        showline=True,
        zeroline=False,
        showgrid=False,
        type='category'
    )
    
    fig.update_yaxes(
        ticktext=pivot_df.index.tolist(),
        tickvals=y,
        title=dict(
            text='Y',
            font=dict(color='black', size=14)
        ),
        tickfont=dict(color='black', size=12),
        gridcolor='black',
        linecolor='black',
        mirror=True,
        showline=True,
        zeroline=False,
        showgrid=False,
        type='category'
    )
    
    fig.update_layout(
        title='Contour Map',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=100, t=50, b=100)
    )
    
    return fig

def create_additional_plots(pivot_df: pd.DataFrame, colorscale: str = 'viridis',
                           x_label: str = "X", y_label: str = "Y",
                           colorbar_title: str = "Value") -> List:
    """Create all additional plots from original code"""
    plots = []
    
    # Handle NaN values
    z_data = pivot_df.values.copy()
    if np.isnan(z_data).any():
        z_data = np.nan_to_num(z_data)
    
    # 1. 3D Surface Plot
    if len(pivot_df.columns) > 1 and len(pivot_df.index) > 1:
        fig_3d = go.Figure(data=go.Surface(
            z=z_data,
            x=list(range(len(pivot_df.columns))),
            y=list(range(len(pivot_df.index))),
            colorscale=colorscale,
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project=dict(z=True)
                )
            ),
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(color='black', size=12)
                ),
                tickfont=dict(color='black', size=10)
            )
        ))

        fig_3d.update_layout(
            title='3D Surface Plot',
            scene=dict(
                xaxis=dict(
                    title=dict(text=x_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.columns.tolist(),
                    tickvals=list(range(len(pivot_df.columns))),
                    tickfont=dict(color='black', size=10)
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.index.tolist(),
                    tickvals=list(range(len(pivot_df.index))),
                    tickfont=dict(color='black', size=10)
                ),
                zaxis=dict(
                    title=dict(text=colorbar_title, font=dict(color='black', size=12)),
                    tickfont=dict(color='black', size=10)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        plots.append(('3D Surface Plot', fig_3d))
    
    # 2. Wireframe Plot
    if len(pivot_df.columns) > 1 and len(pivot_df.index) > 1:
        fig_wire = go.Figure(data=go.Surface(
            z=z_data,
            x=list(range(len(pivot_df.columns))),
            y=list(range(len(pivot_df.index))),
            colorscale=colorscale,
            opacity=0.8,
            showscale=True,
            contours=dict(
                z=dict(show=True, width=1)
            ),
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(color='black', size=12)
                ),
                tickfont=dict(color='black', size=10)
            )
        ))
        
        fig_wire.update_traces(contours_z=dict(show=True, usecolormap=True, project_z=True))

        fig_wire.update_layout(
            title='3D Wireframe Plot',
            scene=dict(
                xaxis=dict(
                    title=dict(text=x_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.columns.tolist(),
                    tickvals=list(range(len(pivot_df.columns))),
                    tickfont=dict(color='black', size=10)
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.index.tolist(),
                    tickvals=list(range(len(pivot_df.index))),
                    tickfont=dict(color='black', size=10)
                ),
                zaxis=dict(
                    title=dict(text=colorbar_title, font=dict(color='black', size=12)),
                    tickfont=dict(color='black', size=10)
                )
            ),
            width=800,
            height=600
        )
        plots.append(('3D Wireframe Plot', fig_wire))
    
    # 3. Density Heatmap
    fig_density = go.Figure(data=go.Heatmap(
        z=z_data,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale=colorscale,
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text=colorbar_title,
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black', size=10)
        ),
        xgap=0.5,
        ygap=0.5
    ))
    
    fig_density.update_layout(
        title='Density Heatmap',
        xaxis=dict(
            title=dict(
                text=x_label,
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12),
            gridcolor='black',
            linecolor='black',
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False,
            tickangle=45
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12),
            gridcolor='black',
            linecolor='black',
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600
    )
    plots.append(('Density Heatmap', fig_density))
    
    # 4. Gradient Vector Field
    if len(pivot_df.columns) > 2 and len(pivot_df.index) > 2 and not np.isnan(z_data).any():
        try:
            grad_y, grad_x = np.gradient(z_data)
            X, Y = np.meshgrid(np.arange(len(pivot_df.columns)), np.arange(len(pivot_df.index)))
            
            fig_gradient = go.Figure()
            
            # Heatmap background
            fig_gradient.add_trace(go.Heatmap(
                z=z_data,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale=colorscale,
                showscale=True,
                opacity=0.7,
                colorbar=dict(
                    title=dict(
                        text=colorbar_title,
                        font=dict(color='black', size=12)
                    ),
                    tickfont=dict(color='black', size=10)
                )
            ))
            
            # Gradient vectors
            skip = max(1, len(pivot_df.columns) // 10)
            for i in range(0, len(pivot_df.index), skip):
                for j in range(0, len(pivot_df.columns), skip):
                    if not (np.isnan(grad_x[i, j]) or np.isnan(grad_y[i, j])):
                        fig_gradient.add_trace(go.Scatter(
                            x=[j, j + grad_x[i, j] * 0.3],
                            y=[i, i + grad_y[i, j] * 0.3],
                            mode='lines',
                            line=dict(color='white', width=2),
                            showlegend=False
                        ))
                        fig_gradient.add_trace(go.Scatter(
                            x=[j + grad_x[i, j] * 0.3],
                            y=[i + grad_y[i, j] * 0.3],
                            mode='markers',
                            marker=dict(color='white', size=5, symbol='triangle-right'),
                            showlegend=False
                        ))
            
            fig_gradient.update_layout(
                title='Gradient Field Overlay',
                xaxis=dict(
                    title=x_label,
                    tickfont=dict(color='black', size=12),
                    gridcolor='black',
                    linecolor='black',
                    mirror=True,
                    showline=True,
                    tickangle=45
                ),
                yaxis=dict(
                    title=y_label,
                    tickfont=dict(color='black', size=12),
                    gridcolor='black',
                    linecolor='black',
                    mirror=True,
                    showline=True
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                width=800,
                height=600,
                showlegend=False
            )
            plots.append(('Gradient Field', fig_gradient))
        except:
            pass
    
    return plots

# ==================== ADVANCED VISUALIZATION FUNCTIONS ====================
def create_marginal_heatmap(pivot_df: pd.DataFrame, colorscale: str = 'viridis',
                           x_label: str = "X", y_label: str = "Y",
                           colorbar_title: str = "Value",
                           show_row_stats: bool = True,
                           show_col_stats: bool = True,
                           show_dendrograms: bool = False,
                           cluster_data: bool = False,
                           n_clusters: int = 3):
    """Create heatmap with marginal plots (ggside-like functionality)"""
    
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Calculate statistics
    stats_data = calculate_statistics(pivot_df)
    
    # Perform clustering if requested
    cluster_labels = None
    if cluster_data:
        cluster_labels = perform_clustering(pivot_df, n_clusters)
    
    # Create subplot grid
    if show_dendrograms:
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.85, 0.15],
            row_heights=[0.15, 0.85],
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
            shared_xaxes=True,
            shared_yaxes=True
        )
    else:
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.85, 0.15],
            row_heights=[0.15, 0.85],
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
            shared_xaxes=True,
            shared_yaxes=True
        )
    
    # Main heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(color='black', size=12)
                ),
                tickfont=dict(color='black', size=10),
                orientation='v',
                x=1.02
            ),
            xgap=1,
            ygap=1
        ),
        row=2, col=1
    )
    
    # Row statistics (right side)
    if show_row_stats and stats_data is not None:
        fig.add_trace(
            go.Bar(
                x=stats_data['row_means'].values,
                y=pivot_df.index.tolist(),
                orientation='h',
                marker=dict(color='lightblue', line=dict(color='black', width=1)),
                name='Row Means',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Column statistics (top side)
    if show_col_stats and stats_data is not None:
        fig.add_trace(
            go.Bar(
                x=pivot_df.columns.tolist(),
                y=stats_data['col_means'].values,
                marker=dict(color='lightcoral', line=dict(color='black', width=1)),
                name='Column Means',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Heatmap with Marginal Statistics",
            font=dict(size=18, color='black'),
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=700,
        showlegend=False,
        margin=dict(l=50, r=100, t=50, b=100)
    )
    
    # Update axes
    fig.update_xaxes(
        row=2, col=1,
        title=dict(text=x_label, font=dict(color='black', size=14)),
        tickfont=dict(color='black', size=12),
        showgrid=False,
        tickangle=45
    )
    
    fig.update_yaxes(
        row=2, col=1,
        title=dict(text=y_label, font=dict(color='black', size=14)),
        tickfont=dict(color='black', size=12),
        showgrid=False
    )
    
    # Hide axes for marginal plots
    fig.update_xaxes(row=1, col=1, showticklabels=False, title_text="")
    fig.update_yaxes(row=2, col=2, showticklabels=False, title_text="")
    
    return fig

def create_statistical_heatmap(pivot_df: pd.DataFrame, colorscale: str = 'viridis',
                              statistical_method: str = 'pvalue'):
    """Create heatmap with statistical annotations"""
    
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Create figure with statistical annotations
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale=colorscale,
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text=f'Value ({statistical_method})',
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black', size=10)
        )
    ))
    
    # Add statistical annotations if needed
    if statistical_method == 'zscore':
        # Calculate z-scores
        z_scores = (pivot_df - pivot_df.mean()) / pivot_df.std()
        # Add annotation for significant z-scores
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                if abs(z_scores.iloc[i, j]) > 2:
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text="*",
                        showarrow=False,
                        font=dict(color='white', size=14)
                    )
    
    fig.update_layout(
        title=dict(
            text=f"Statistical Heatmap ({statistical_method})",
            font=dict(size=16, color='black'),
            x=0.5
        ),
        xaxis=dict(
            tickfont=dict(color='black', size=12),
            title_font=dict(color='black', size=14),
            tickangle=45
        ),
        yaxis=dict(
            tickfont=dict(color='black', size=12),
            title_font=dict(color='black', size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=100, t=50, b=100)
    )
    
    return fig

def create_clustered_heatmap(pivot_df: pd.DataFrame, colorscale: str = 'viridis',
                            n_clusters: int = 3, show_dendrogram: bool = True):
    """Create heatmap with clustering"""
    
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Perform clustering
    cluster_labels = perform_clustering(pivot_df, n_clusters)
    
    if cluster_labels is None:
        return create_basic_heatmap(pivot_df, colorscale)
    
    # Sort data by cluster labels
    sorted_indices = np.argsort(cluster_labels)
    sorted_data = pivot_df.iloc[sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sorted_data.values,
        x=sorted_data.columns.tolist(),
        y=sorted_data.index.tolist(),
        colorscale=colorscale,
        colorbar=dict(
            title=dict(
                text='Value',
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black', size=10)
        )
    ))
    
    # Add cluster annotations
    unique_clusters = np.unique(sorted_labels)
    colors = px.colors.qualitative.Set3[:len(unique_clusters)]
    
    # Add cluster annotations on the side
    current_idx = 0
    for cluster_id in unique_clusters:
        cluster_size = np.sum(sorted_labels == cluster_id)
        if cluster_size > 0:
            fig.add_annotation(
                x=-0.5,
                y=current_idx + cluster_size/2,
                text=f"Cluster {int(cluster_id) + 1}",
                showarrow=False,
                font=dict(color='black', size=12, family="Arial Black"),
                bgcolor=colors[int(cluster_id) % len(colors)],
                opacity=0.8,
                xref="x",
                yref="y"
            )
            current_idx += cluster_size
    
    fig.update_layout(
        title=dict(
            text=f"Clustered Heatmap (k={n_clusters})",
            font=dict(size=16, color='black'),
            x=0.5
        ),
        xaxis=dict(
            tickfont=dict(color='black', size=12),
            title_font=dict(color='black', size=14),
            tickangle=45
        ),
        yaxis=dict(
            tickfont=dict(color='black', size=12),
            title_font=dict(color='black', size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=600,
        margin=dict(l=100, r=100, t=50, b=100)
    )
    
    return fig

def create_correlation_heatmap(pivot_df: pd.DataFrame, colorscale: str = 'RdBu'):
    """Create correlation matrix heatmap"""
    
    if pivot_df is None:
        return None
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlations(pivot_df)
    
    if corr_matrix is None:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(
            title=dict(
                text='Correlation',
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black', size=10)
        )
    ))
    
    # Add correlation values as text
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{corr_matrix.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(
                    color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                    size=10
                )
            )
    
    fig.update_layout(
        title=dict(
            text="Correlation Matrix",
            font=dict(size=16, color='black'),
            x=0.5
        ),
        xaxis=dict(
            tickfont=dict(color='black', size=12),
            tickangle=45
        ),
        yaxis=dict(
            tickfont=dict(color='black', size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=50, r=100, t=50, b=100)
    )
    
    return fig

def create_time_series_heatmap(pivot_df: pd.DataFrame, colorscale: str = 'viridis'):
    """Create heatmap for time series data"""
    
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Create subplots for heatmap and trends
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=('Time Series Heatmap', 'Sample Trends'),
        horizontal_spacing=0.1
    )
    
    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            colorscale=colorscale,
            colorbar=dict(
                title=dict(
                    text='Value',
                    font=dict(color='black', size=12)
                ),
                tickfont=dict(color='black', size=10)
            )
        ),
        row=1, col=1
    )
    
    # Trend lines for selected rows
    n_samples = min(5, len(pivot_df))
    sample_indices = np.linspace(0, len(pivot_df)-1, n_samples, dtype=int)
    
    for idx in sample_indices:
        row_data = pivot_df.iloc[idx].values
        row_name = pivot_df.index[idx]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(row_data))),
                y=row_data,
                mode='lines+markers',
                name=f'{row_name}',
                line=dict(width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=dict(
            text="Time Series Analysis",
            font=dict(size=16, color='black'),
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=600,
        showlegend=True,
        margin=dict(l=50, r=100, t=50, b=100)
    )
    
    fig.update_xaxes(title_text="Time Points", row=1, col=1, 
                    tickfont=dict(color='black', size=12),
                    tickangle=45)
    fig.update_yaxes(title_text="Samples", row=1, col=1, 
                    tickfont=dict(color='black', size=12))
    fig.update_xaxes(title_text="Time", row=1, col=2, 
                    tickfont=dict(color='black', size=12))
    fig.update_yaxes(title_text="Value", row=1, col=2, 
                    tickfont=dict(color='black', size=12))
    
    return fig

# ==================== EXPORT FUNCTIONS ====================
def save_all_plots_matplotlib(pivot_df, normalized_df, x_label, y_label, 
                             colorbar_title, additional_plots, dpi=300, 
                             show_values=True):
    """Save all plots as high-resolution PNG files"""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        plot_counter = 1
        
        # 1. Main Heatmap
        fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
        plot_data = pivot_df.values.copy()
        if np.isnan(plot_data).any():
            plot_data = np.nan_to_num(plot_data)
        
        im = ax.imshow(plot_data, aspect='auto', cmap='viridis',
                      extent=[0, len(pivot_df.columns), 0, len(pivot_df.index)])
        
        ax.set_xticks(np.arange(len(pivot_df.columns)) + 0.5)
        ax.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
        ax.set_xticklabels(pivot_df.columns.tolist())
        ax.set_yticklabels(pivot_df.index.tolist())
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label(colorbar_title, rotation=270, labelpad=25, fontsize=14, color='black')
        cbar.ax.tick_params(colors='black', labelsize=12)
        cbar.ax.yaxis.label.set_color('black')
        
        ax.set_xlabel(x_label, fontsize=16, color='black', labelpad=15)
        ax.set_ylabel(y_label, fontsize=16, color='black', labelpad=15)
        ax.set_title('Main Heatmap', fontsize=18, color='black', pad=20)
        
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='x', colors='black', labelsize=12)
        ax.tick_params(axis='y', colors='black', labelsize=12)
        
        if show_values:
            for i in range(len(pivot_df.index)):
                for j in range(len(pivot_df.columns)):
                    value = pivot_df.values[i, j]
                    if not np.isnan(value):
                        ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                              ha="center", va="center", 
                              color="white" if value > np.nanmean(plot_data) else "black",
                              fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr(f'{plot_counter:02d}_main_heatmap.png', buffer.getvalue())
        plt.close(fig)
        plot_counter += 1
        
        # 2. Normalized Heatmap
        if normalized_df is not None:
            fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
            norm_data = normalized_df.values.copy()
            if np.isnan(norm_data).any():
                norm_data = np.nan_to_num(norm_data)
            
            im = ax.imshow(norm_data, aspect='auto', cmap='viridis',
                          extent=[0, len(normalized_df.columns), 0, len(normalized_df.index)])
            
            ax.set_xticks(np.arange(len(normalized_df.columns)) + 0.5)
            ax.set_yticks(np.arange(len(normalized_df.index)) + 0.5)
            ax.set_xticklabels(normalized_df.columns.tolist())
            ax.set_yticklabels(normalized_df.index.tolist())
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            cbar = plt.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label(f'{colorbar_title} (normalized)', rotation=270, 
                          labelpad=25, fontsize=14, color='black')
            cbar.ax.tick_params(colors='black', labelsize=12)
            
            ax.set_xlabel(x_label, fontsize=16, color='black', labelpad=15)
            ax.set_ylabel(y_label, fontsize=16, color='black', labelpad=15)
            ax.set_title('Normalized Heatmap', fontsize=18, color='black', pad=20)
            
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.tick_params(axis='x', colors='black', labelsize=12)
            ax.tick_params(axis='y', colors='black', labelsize=12)
            
            if show_values:
                for i in range(len(normalized_df.index)):
                    for j in range(len(normalized_df.columns)):
                        value = normalized_df.values[i, j]
                        if not np.isnan(value):
                            ax.text(j + 0.5, i + 0.5, f'{value:.3f}',
                                  ha="center", va="center", 
                                  color="white" if value > 0.5 else "black",
                                  fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
            zip_file.writestr(f'{plot_counter:02d}_normalized_heatmap.png', buffer.getvalue())
            plt.close(fig)
            plot_counter += 1
        
        # 3. Contour Plot
        fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
        contour_data = pivot_df.values.copy()
        if np.isnan(contour_data).any():
            contour_data = np.nan_to_num(contour_data)
        
        X, Y = np.meshgrid(np.arange(len(pivot_df.columns)) + 0.5, 
                          np.arange(len(pivot_df.index)) + 0.5)
        
        contour = ax.contourf(X, Y, contour_data, cmap='viridis', levels=20)
        ax.contour(X, Y, contour_data, colors='black', linewidths=0.5, levels=10)
        
        ax.set_xticks(np.arange(len(pivot_df.columns)) + 0.5)
        ax.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
        ax.set_xticklabels(pivot_df.columns.tolist())
        ax.set_yticklabels(pivot_df.index.tolist())
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
        cbar.set_label(colorbar_title, rotation=270, labelpad=25, fontsize=14, color='black')
        cbar.ax.tick_params(colors='black', labelsize=12)
        
        ax.set_xlabel(x_label, fontsize=16, color='black', labelpad=15)
        ax.set_ylabel(y_label, fontsize=16, color='black', labelpad=15)
        ax.set_title('Contour Plot', fontsize=18, color='black', pad=20)
        
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='x', colors='black', labelsize=12)
        ax.tick_params(axis='y', colors='black', labelsize=12)
        
        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr(f'{plot_counter:02d}_contour_plot.png', buffer.getvalue())
        plt.close(fig)
        plot_counter += 1
        
        # Additional plots
        for plot_name, plot_fig in additional_plots:
            if plot_fig is not None:
                try:
                    # Convert plotly to image
                    if hasattr(plot_fig, 'to_image'):
                        img_bytes = plot_fig.to_image(format="png", width=1600, height=1200)
                        zip_file.writestr(f'{plot_counter:02d}_{plot_name.lower().replace(" ", "_")}.png', img_bytes)
                        plot_counter += 1
                except:
                    continue
        
        # Data table
        fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for i in range(min(50, len(pivot_df.index))):  # Limit rows for readability
            row = []
            for j in range(min(20, len(pivot_df.columns))):  # Limit columns
                value = pivot_df.values[i, j]
                row.append(f'{value:.3f}' if not np.isnan(value) else 'NaN')
            table_data.append(row)
        
        row_labels = pivot_df.index.tolist()[:50]
        col_labels = pivot_df.columns.tolist()[:20]
        
        table = ax.table(cellText=table_data,
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        for key, cell in table.get_celld().items():
            cell.set_text_props(color='black')
            if key[0] == 0:  # Header
                cell.set_facecolor('#4B8BBE')
                cell.set_text_props(color='white', weight='bold')
            elif key[1] == -1:  # Row labels
                cell.set_facecolor('#F0F0F0')
        
        ax.set_title('Data Table (First 50 rows × 20 columns)', fontsize=16, pad=20, color='black')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr(f'{plot_counter:02d}_data_table.png', buffer.getvalue())
        plt.close(fig)
    
    return zip_buffer

# ==================== MAIN APPLICATION ====================
st.title("🔥 Advanced Scientific Heatmap Generator")
st.markdown("""
Generate publication-ready heatmaps with advanced statistical analysis and visualization features.
All features are optional and can be customized for your specific needs.
""")

# Initialize session state
if 'all_plots' not in st.session_state:
    st.session_state.all_plots = {}
if 'additional_plots' not in st.session_state:
    st.session_state.additional_plots = []

# Sidebar with all settings
with st.sidebar:
    st.header("⚙️ Core Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        x_label = st.text_input("X-axis label", value="X")
    with col2:
        y_label = st.text_input("Y-axis label", value="Y")
    
    colorbar_title = st.text_input("Colorbar title", value="Value")
    
    # Data processing
    st.subheader("📊 Data Processing")
    sort_numerically = st.checkbox("Sort axes numerically", value=True)
    show_values = st.checkbox("Show values in cells", value=False)
    
    value_format = st.selectbox(
        "Value format",
        [".0f", ".1f", ".2f", ".3f", ".2e"]
    )
    
    # Color settings
    st.subheader("🎨 Color Settings")
    builtin_palettes = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis",
                       "RdBu", "RdYlBu", "Rainbow", "Portland", "Jet",
                       "Greys", "Hot", "Electric", "Blues", "Greens"]
    selected_palette = st.selectbox("Color palette", builtin_palettes, index=0)
    
    # Normalization
    normalization_method = st.selectbox(
        "Normalization method",
        ["None", "minmax", "zscore", "log10", "percentile", "robust"]
    )
    
    # Original plots selection
    st.header("📈 Original Plots")
    show_normalized = st.checkbox("Show normalized heatmap", value=True)
    show_contour = st.checkbox("Show contour map", value=True)
    show_additional = st.checkbox("Show additional plots (3D, Gradient, etc.)", value=True)
    
    if show_contour:
        st.subheader("Contour Settings")
        contour_smoothing = st.slider("Smoothing level", 0.0, 3.0, 1.0, 0.1)
        show_contour_lines = st.checkbox("Show contour lines", value=True)
        show_contour_labels = st.checkbox("Show contour labels", value=True)
    
    # Advanced features
    st.header("🔬 Advanced Features")
    
    # GGside-like features
    st.subheader("📊 Marginal Plots (ggside)")
    show_marginal = st.checkbox("Show marginal statistics", value=False)
    if show_marginal:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            show_row_stats = st.checkbox("Row stats", value=True)
        with col_m2:
            show_col_stats = st.checkbox("Column stats", value=True)
        cluster_marginal = st.checkbox("Cluster marginal data", value=False)
        if cluster_marginal:
            n_clusters_m = st.slider("Clusters", 2, 10, 3, key='marginal')
    
    # Statistical analysis
    st.subheader("📊 Statistical Analysis")
    show_statistical = st.checkbox("Show statistical heatmap", value=False)
    if show_statistical:
        stat_method = st.selectbox(
            "Method",
            ["zscore", "percentile", "pvalue"],
            key='stat'
        )
    
    # Clustering
    st.subheader("🗂️ Clustering")
    show_clustered = st.checkbox("Show clustered heatmap", value=False)
    if show_clustered:
        n_clusters = st.slider("Number of clusters", 2, 10, 3, key='cluster')
        cluster_method = st.selectbox(
            "Clustering method",
            ["kmeans", "hierarchical"]
        )
    
    # Correlation
    st.subheader("📈 Correlation Analysis")
    show_correlation = st.checkbox("Show correlation matrix", value=False)
    
    # Time series
    st.subheader("⏰ Time Series")
    show_time_series = st.checkbox("Show time series analysis", value=False)
    
    # Export settings
    st.header("💾 Export Settings")
    save_dpi = st.slider("DPI for export", 100, 600, 300)
    
    # Reset button
    if st.button("🔄 Reset All Settings"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content area
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("📥 Data Input")
    
    example_choice = st.selectbox(
        "Choose example data or upload your own",
        ["Upload your own data", 
         "Example 1: Simple Matrix",
         "Example 2: Gene Expression", 
         "Example 3: Time Series Data",
         "Example 4: Clinical Measurements"]
    )
    
    if example_choice == "Example 1: Simple Matrix":
        example_data = """A,Jan,10
A,Feb,20
B,Jan,15
B,Feb,25
C,Jan,30
C,Feb,35"""
    elif example_choice == "Example 2: Gene Expression":
        example_data = """Gene,T0,T1,T2,T3,T4
GeneA,1.2,2.3,3.1,2.8,1.9
GeneB,0.8,1.5,2.2,2.1,1.3
GeneC,2.1,3.8,4.5,4.2,3.1
GeneD,1.5,2.8,3.9,3.5,2.4
GeneE,0.9,1.8,2.7,2.4,1.5"""
    elif example_choice == "Example 3: Time Series Data":
        example_data = """Time,Sample1,Sample2,Sample3,Sample4,Sample5
0h,10.2,12.5,9.8,11.3,10.7
1h,15.3,18.2,14.7,16.8,15.9
2h,22.1,25.4,21.3,23.7,22.5
3h,18.7,21.3,17.9,19.8,18.9
4h,12.8,15.1,11.9,13.7,12.9
5h,8.5,10.2,7.9,9.3,8.7"""
    elif example_choice == "Example 4: Clinical Measurements":
        example_data = """Patient,BP_Systolic,BP_Diastolic,Heart_Rate,Temperature,Glucose
P001,120,80,72,36.6,5.2
P002,135,85,68,36.8,6.1
P003,118,78,75,36.5,5.8
P004,142,92,80,37.1,7.2
P005,128,84,70,36.7,5.5
P006,130,82,73,36.9,6.3"""
    else:
        example_data = ""
    
    data_input = st.text_area(
        "Enter your data (CSV, TSV, or space-separated):",
        value=example_data,
        height=250,
        help="Format: X,Y,Value or with headers. Each row should contain X, Y, and Value."
    )
    
    uploaded_file = st.file_uploader(
        "📁 Or upload a data file",
        type=['txt', 'csv', 'tsv', 'xlsx', 'xls'],
        help="Supported formats: CSV, TSV, Excel, Text"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                df_uploaded = pd.read_excel(uploaded_file)
                data_input = df_uploaded.to_csv(index=False)
            else:
                content = uploaded_file.read().decode('utf-8')
                data_input = content
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    generate_button = st.button(
        "🚀 Generate All Selected Plots", 
        type="primary", 
        use_container_width=True,
        help="Generate all selected visualizations based on current settings"
    )

with col_right:
    st.header("📋 Data Preview")
    
    if generate_button and data_input.strip():
        with st.spinner("🔄 Processing data and generating plots..."):
            # Parse data
            df = parse_data(data_input)
            
            if df is not None and not df.empty:
                st.session_state.df = df
                st.session_state.data_ready = True
                
                # Create pivot table
                pivot_df = create_pivot_table(df, sort_numerically)
                
                if pivot_df is not None:
                    st.session_state.pivot_df = pivot_df
                    
                    # Apply normalization
                    normalized_df = None
                    if normalization_method != "None":
                        normalized_df = normalize_data(pivot_df, normalization_method)
                    else:
                        normalized_df = pivot_df
                    
                    st.session_state.normalized_df = normalized_df
                    
                    # Generate all requested plots
                    all_plots = {}
                    additional_plots_list = []
                    
                    # 1. Basic Heatmap
                    fig_basic = create_basic_heatmap(
                        normalized_df, selected_palette, x_label, y_label,
                        colorbar_title, show_values, value_format
                    )
                    if fig_basic:
                        all_plots["Basic Heatmap"] = fig_basic
                    
                    # 2. Normalized Heatmap
                    if show_normalized and normalization_method != "None":
                        fig_norm = create_normalized_heatmap(
                            normalized_df, selected_palette, x_label, y_label,
                            f"{colorbar_title} ({normalization_method})", normalization_method
                        )
                        if fig_norm:
                            all_plots["Normalized Heatmap"] = fig_norm
                    
                    # 3. Contour Map
                    if show_contour:
                        fig_contour = create_smooth_contour(
                            normalized_df, selected_palette, contour_smoothing,
                            show_contour_lines, show_contour_labels
                        )
                        if fig_contour:
                            all_plots["Contour Map"] = fig_contour
                    
                    # 4. Additional plots from original code
                    if show_additional:
                        original_additional = create_additional_plots(
                            normalized_df, selected_palette, x_label, y_label, colorbar_title
                        )
                        for plot_name, plot_fig in original_additional:
                            if plot_fig:
                                all_plots[plot_name] = plot_fig
                                additional_plots_list.append((plot_name, plot_fig))
                    
                    # 5. Marginal Heatmap (ggside-like)
                    if show_marginal:
                        fig_marginal = create_marginal_heatmap(
                            normalized_df, selected_palette, x_label, y_label,
                            colorbar_title, show_row_stats, show_col_stats,
                            False, cluster_marginal if 'cluster_marginal' in locals() else False,
                            n_clusters_m if 'n_clusters_m' in locals() else 3
                        )
                        if fig_marginal:
                            all_plots["Marginal Heatmap"] = fig_marginal
                    
                    # 6. Statistical Heatmap
                    if show_statistical:
                        fig_stat = create_statistical_heatmap(
                            normalized_df, selected_palette, stat_method
                        )
                        if fig_stat:
                            all_plots["Statistical Heatmap"] = fig_stat
                    
                    # 7. Clustered Heatmap
                    if show_clustered:
                        fig_clust = create_clustered_heatmap(
                            normalized_df, selected_palette, n_clusters
                        )
                        if fig_clust:
                            all_plots["Clustered Heatmap"] = fig_clust
                    
                    # 8. Correlation Matrix
                    if show_correlation:
                        fig_corr = create_correlation_heatmap(pivot_df, 'RdBu')
                        if fig_corr:
                            all_plots["Correlation Matrix"] = fig_corr
                    
                    # 9. Time Series Analysis
                    if show_time_series:
                        fig_time = create_time_series_heatmap(normalized_df, selected_palette)
                        if fig_time:
                            all_plots["Time Series Analysis"] = fig_time
                    
                    st.session_state.all_plots = all_plots
                    st.session_state.additional_plots = additional_plots_list
                    
                    st.success("✅ All plots generated successfully!")
                    
                    # Display data statistics
                    st.subheader("📊 Data Statistics")
                    stats_data = calculate_statistics(pivot_df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Rows", len(pivot_df))
                        st.metric("Row Mean", f"{stats_data['row_means'].mean():.3f}")
                    
                    with col2:
                        st.metric("Columns", len(pivot_df.columns))
                        st.metric("Column Mean", f"{stats_data['col_means'].mean():.3f}")
                    
                    with col3:
                        st.metric("Global Mean", f"{stats_data['global_mean']:.3f}")
                        st.metric("Global Std", f"{stats_data['global_std']:.3f}")
                    
                    with col4:
                        st.metric("Min Value", f"{stats_data['global_min']:.3f}")
                        st.metric("Max Value", f"{stats_data['global_max']:.3f}")
                    
                    # Display data preview
                    with st.expander("📄 View Full Data Table"):
                        st.dataframe(pivot_df, use_container_width=True)
                    
            else:
                st.error("❌ Failed to process data. Please check the format.")

# Display generated plots
if 'all_plots' in st.session_state and st.session_state.all_plots:
    st.markdown("---")
    st.header("📊 Generated Visualizations")
    
    # Create tabs for different plot categories
    tabs = st.tabs(["Basic Plots", "Advanced Analysis", "3D & Special", "Export"])
    
    with tabs[0]:
        st.subheader("Basic Heatmaps")
        
        if "Basic Heatmap" in st.session_state.all_plots:
            st.plotly_chart(st.session_state.all_plots["Basic Heatmap"], 
                          use_container_width=True)
        
        if "Normalized Heatmap" in st.session_state.all_plots:
            st.plotly_chart(st.session_state.all_plots["Normalized Heatmap"], 
                          use_container_width=True)
        
        if "Contour Map" in st.session_state.all_plots:
            st.plotly_chart(st.session_state.all_plots["Contour Map"], 
                          use_container_width=True)
    
    with tabs[1]:
        st.subheader("Advanced Statistical Analysis")
        
        cols = st.columns(2)
        plot_idx = 0
        
        for plot_name, plot_fig in st.session_state.all_plots.items():
            if plot_name in ["Marginal Heatmap", "Statistical Heatmap", 
                           "Clustered Heatmap", "Correlation Matrix", 
                           "Time Series Analysis"]:
                with cols[plot_idx % 2]:
                    st.plotly_chart(plot_fig, use_container_width=True)
                plot_idx += 1
    
    with tabs[2]:
        st.subheader("3D and Special Visualizations")
        
        for plot_name, plot_fig in st.session_state.all_plots.items():
            if plot_name in ["3D Surface Plot", "3D Wireframe Plot", 
                           "Density Heatmap", "Gradient Field"]:
                st.plotly_chart(plot_fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("📦 Export All Plots")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("💾 Download All Plots (ZIP)", use_container_width=True):
                try:
                    with st.spinner("Creating high-resolution ZIP archive..."):
                        zip_buffer = save_all_plots_matplotlib(
                            st.session_state.pivot_df,
                            st.session_state.normalized_df,
                            x_label, y_label, colorbar_title,
                            st.session_state.additional_plots,
                            save_dpi, show_values
                        )
                        
                        st.download_button(
                            label=f"⬇️ Download ZIP ({save_dpi} DPI)",
                            data=zip_buffer.getvalue(),
                            file_name=f"scientific_heatmaps_{save_dpi}dpi.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
        
        with col_exp2:
            if 'df' in st.session_state:
                csv_data = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="📄 Export Raw Data",
                    data=csv_data,
                    file_name="heatmap_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp3:
            if 'pivot_df' in st.session_state:
                pivot_csv = st.session_state.pivot_df.to_csv()
                st.download_button(
                    label="📊 Export Pivot Table",
                    data=pivot_csv,
                    file_name="pivot_table.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Plot summary
        st.subheader("Generated Plots Summary")
        plot_list = list(st.session_state.all_plots.keys())
        
        cols_summary = st.columns(3)
        for i, plot_name in enumerate(plot_list):
            with cols_summary[i % 3]:
                st.info(f"**{i+1}. {plot_name}**")

# Documentation
with st.expander("📚 User Guide & Documentation"):
    st.markdown("""
    ## 🎯 How to Use This Tool
    
    ### 1. Data Input
    - **Format**: CSV, TSV, or space-separated values
    - **Structure**: Each row should contain X, Y, and Value
    - **Headers**: Optional, can be included in first row
    
    ### 2. Core Features
    
    #### Basic Plots:
    - **Heatmap**: Standard heatmap visualization
    - **Normalized Heatmap**: Data normalized using selected method
    - **Contour Map**: Smooth contour visualization
    - **3D Plots**: Surface and wireframe 3D visualizations
    
    #### Advanced Features (Optional):
    - **Marginal Plots**: Statistics on heatmap margins (ggside-like)
    - **Statistical Heatmaps**: Z-scores, p-values, significance markers
    - **Clustering**: K-means and hierarchical clustering
    - **Correlation Analysis**: Correlation matrix heatmap
    - **Time Series**: Temporal pattern analysis
    
    ### 3. Customization Options
    
    #### Color Settings:
    - 15 built-in color palettes
    - Optimized for scientific publications
    - Colorblind-friendly options available
    
    #### Display Options:
    - Show/hide cell values
    - Format values (integer, decimal, scientific)
    - Adjust font sizes and labels
    - Custom axis labels
    
    ### 4. Export Options
    
    #### High-Resolution Export:
    - All plots as PNG images (100-600 DPI)
    - Publication-ready formatting
    - Black axis labels (meets journal requirements)
    - Organized ZIP file with numbered files
    
    #### Data Export:
    - Raw data as CSV
    - Pivot table as CSV
    - Statistical summary
    
    ### 5. Scientific Applications
    
    #### Bioinformatics:
    - Gene expression heatmaps
    - Clustering of genes/samples
    - Statistical significance
    
    #### Clinical Research:
    - Patient data visualization
    - Treatment response patterns
    - Correlation analysis
    
    #### Time Series Analysis:
    - Temporal pattern visualization
    - Trend analysis
    - Comparative studies
    
    ### 6. Tips for Publication
    
    1. **Choose appropriate color palette**: Use sequential palettes for continuous data
    2. **Enable value display**: For precise data presentation
    3. **Use normalization**: When comparing datasets with different scales
    4. **Add statistical annotations**: For hypothesis testing results
    5. **Export at 300+ DPI**: For print publications
    
    ### 7. Troubleshooting
    
    - **Data format errors**: Ensure consistent delimiter usage
    - **Missing values**: Tool handles NaN values automatically
    - **Large datasets**: Consider subset for initial visualization
    - **Memory issues**: Reduce number of simultaneous plots
    
    ### 8. Citation & Attribution
    
    When using this tool for publications, please cite:
    > Advanced Scientific Heatmap Generator v2.0
    > [Your Institution/Name], [Year]
    
    ---
    
    **Need help?** Check the examples or contact support.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
    <b>🔥 Advanced Scientific Heatmap Generator v2.0</b><br>
    <small>Designed for scientific publications • All features optional • High-resolution export</small><br>
    <small>Supports: 📊 Basic Heatmaps • 🔬 Statistical Analysis • 🗂️ Clustering • 📈 Time Series • 💾 Publication Export</small>
    </div>
    """,
    unsafe_allow_html=True
)
