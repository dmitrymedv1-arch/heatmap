import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import re
import zipfile
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Optional, Tuple
from scipy.ndimage import gaussian_filter

# Page configuration
st.set_page_config(
    page_title="Heatmap Generator",
    page_icon="🔥",
    layout="wide"
)

# Data processing functions
def preprocess_uploaded_content(content: str) -> str:
    """
    Preprocess uploaded content for handling incomplete cases
    """
    lines = content.strip().split('\n')
    processed_lines = []
    last_x_value = None
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Remove extra spaces
        line = line.strip()
        
        # Split the line into parts
        if '\t' in line:
            parts = line.split('\t')
        elif ',' in line:
            parts = line.split(',')
        else:
            # Split by multiple spaces
            parts = re.split(r'\s+', line)
        
        # Remove empty elements
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 3:
            # Complete string with X, Y and Value
            processed_lines.append(f"{parts[0]},{parts[1]},{parts[2]}")
            last_x_value = parts[0]
        elif len(parts) == 2:
            # Handle cases with missing X values
            if last_x_value is not None:
                processed_lines.append(f"{last_x_value},{parts[0]},{parts[1]}")
            else:
                # If X is not previously defined, use first value as X
                processed_lines.append(f"{parts[0]},{parts[1]},NaN")
        elif len(parts) == 1:
            # Single value - treat as continuation
            continue
    
    return '\n'.join(processed_lines)

def parse_data(content: str) -> pd.DataFrame:
    """
    Parse data from string to DataFrame with robust type handling
    """
    # Preprocess data
    processed_content = preprocess_uploaded_content(content)
    
    # Try to parse as CSV with different delimiters
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
            # Check if we have at least 3 columns
            if df.shape[1] >= 3:
                df = df.iloc[:, :3]
                df.columns = ['X', 'Y', 'Value']
                break
        except:
            continue
    
    if df is None or df.empty:
        st.error("Failed to parse data. Please check the format.")
        return None
    
    # Keep original string representation for X and Y
    df['X'] = df['X'].astype(str)
    df['Y'] = df['Y'].astype(str)
    
    # Try to convert Value to numeric, preserving non-numeric values
    try:
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        if df['Value'].isna().any():
            st.warning("Some values could not be converted to numbers. These will be treated as NaN.")
    except Exception as e:
        st.warning(f"Could not convert values to numeric format: {e}")
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Create numeric versions for sorting if possible
    df['X_numeric'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y_numeric'] = pd.to_numeric(df['Y'], errors='coerce')
    
    # Remove rows with NaN values in X or Y
    df = df.dropna(subset=['X', 'Y'])
    
    return df

def create_pivot_table(df: pd.DataFrame, sort_numerically: bool = True) -> pd.DataFrame:
    """
    Create pivot table for heatmap
    """
    if df is None or df.empty:
        return None
    
    # Create pivot table
    pivot_df = df.pivot_table(index='Y', columns='X', values='Value', aggfunc='mean')
    
    # Apply sorting if requested and possible
    if sort_numerically:
        # Try to sort X columns numerically
        try:
            # Extract unique X values with their numeric equivalents
            x_values = df[['X', 'X_numeric']].drop_duplicates().dropna(subset=['X_numeric'])
            if not x_values.empty:
                x_sorted = x_values.sort_values('X_numeric')['X'].tolist()
                # Keep only columns that exist in pivot_df
                x_sorted = [x for x in x_sorted if x in pivot_df.columns]
                # Reorder columns
                pivot_df = pivot_df[x_sorted]
        except:
            pass
        
        # Try to sort Y rows numerically
        try:
            # Extract unique Y values with their numeric equivalents
            y_values = df[['Y', 'Y_numeric']].drop_duplicates().dropna(subset=['Y_numeric'])
            if not y_values.empty:
                y_sorted = y_values.sort_values('Y_numeric')['Y'].tolist()
                # Keep only indices that exist in pivot_df
                y_sorted = [y for y in y_sorted if y in pivot_df.index]
                # Reorder rows
                pivot_df = pivot_df.loc[y_sorted]
        except:
            pass
    
    return pivot_df

def normalize_data(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data (0-1)
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Remove NaN values for min/max calculation
    valid_values = pivot_df.values[~np.isnan(pivot_df.values)]
    
    if len(valid_values) == 0:
        return pivot_df
    
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    
    if max_val == min_val:
        return pivot_df
    
    # Normalization
    normalized_df = (pivot_df - min_val) / (max_val - min_val)
    return normalized_df

def create_smooth_contour(pivot_df: pd.DataFrame, colorscale: str = 'viridis', 
                         smoothing_level: float = 1.0, show_contour_lines: bool = True,
                         show_contour_labels: bool = True) -> go.Figure:
    """
    Create smooth contour plot (height map) with adjustable smoothing
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    # Prepare data for contour plot
    x = list(range(len(pivot_df.columns)))
    y = list(range(len(pivot_df.index)))
    z = pivot_df.values
    
    # Handle NaN values by filling with nearest neighbors
    if np.isnan(z).any():
        from scipy import interpolate
        x_grid, y_grid = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        # Get valid points
        valid_mask = ~np.isnan(z)
        if valid_mask.any():
            interp = interpolate.LinearNDInterpolator(
                list(zip(x_grid[valid_mask], y_grid[valid_mask])), 
                z[valid_mask]
            )
            z = interp(x_grid, y_grid)
        # Fill any remaining NaN with 0
        z = np.nan_to_num(z)
    
    # Apply smoothing if requested
    if smoothing_level > 0:
        z_smoothed = gaussian_filter(z, sigma=smoothing_level)
    else:
        z_smoothed = z
    
    # Create figure with filled contours
    fig = go.Figure(data=go.Contour(
        z=z_smoothed,
        x=x,
        y=y,
        colorscale=colorscale,
        contours=dict(
            showlabels=show_contour_labels,
            labelfont=dict(size=12, color='black'),
            coloring='heatmap',  # Always use heatmap for fill
            showlines=show_contour_lines,
        ),
        line=dict(width=1 if show_contour_lines else 0),
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text='Value',
                font=dict(color='black', size=12)
            ),
            tickfont=dict(color='black')
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
        tickfont=dict(color='black'),
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
        tickfont=dict(color='black'),
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
        width=600,
        height=500
    )
    
    return fig

def create_additional_plots(pivot_df: pd.DataFrame, colorscale: str = 'viridis', 
                           x_label: str = "X", y_label: str = "Y", 
                           colorbar_title: str = "Value") -> List[Tuple[str, go.Figure]]:
    """
    Create additional visualization plots
    """
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
                tickfont=dict(color='black')
            )
        ))

        fig_3d.update_layout(
            title='3D Surface Plot',
            scene=dict(
                xaxis=dict(
                    title=dict(text=x_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.columns.tolist(),
                    tickvals=list(range(len(pivot_df.columns))),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.index.tolist(),
                    tickvals=list(range(len(pivot_df.index))),
                    tickfont=dict(color='black')
                ),
                zaxis=dict(
                    title=dict(text=colorbar_title, font=dict(color='black', size=12)),
                    tickfont=dict(color='black')
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=600,
            height=500,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        plots.append(('3D Surface', fig_3d))
    
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
                tickfont=dict(color='black')
            )
        ))
        
        # Update wireframe appearance
        fig_wire.update_traces(contours_z=dict(show=True, usecolormap=True, project_z=True))

        fig_wire.update_layout(
            title='3D Wireframe Plot',
            scene=dict(
                xaxis=dict(
                    title=dict(text=x_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.columns.tolist(),
                    tickvals=list(range(len(pivot_df.columns))),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(color='black', size=12)),
                    ticktext=pivot_df.index.tolist(),
                    tickvals=list(range(len(pivot_df.index))),
                    tickfont=dict(color='black')
                ),
                zaxis=dict(
                    title=dict(text=colorbar_title, font=dict(color='black', size=12)),
                    tickfont=dict(color='black')
                )
            ),
            width=600,
            height=500
        )
        plots.append(('3D Wireframe', fig_wire))
    
    # 3. Density Heatmap (2D histogram style)
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
            tickfont=dict(color='black')
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
            tickfont=dict(color='black'),
            gridcolor='black',
            linecolor='black',
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black'),
            gridcolor='black',
            linecolor='black',
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=500
    )
    plots.append(('Density Heatmap', fig_density))
    
    # 4. Gradient Vector Field (simplified)
    if len(pivot_df.columns) > 2 and len(pivot_df.index) > 2 and not np.isnan(z_data).any():
        try:
            # Calculate gradients
            grad_y, grad_x = np.gradient(z_data)
            
            # Create quiver plot
            X, Y = np.meshgrid(np.arange(len(pivot_df.columns)), np.arange(len(pivot_df.index)))
            
            fig_gradient = go.Figure()
            
            # Add heatmap as background
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
                    tickfont=dict(color='black')
                )
            ))
            
            # Add gradient vectors (simplified representation)
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
                        # Add arrow head
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
                    tickfont=dict(color='black'),
                    gridcolor='black',
                    linecolor='black',
                    mirror=True,
                    showline=True
                ),
                yaxis=dict(
                    title=y_label,
                    tickfont=dict(color='black'),
                    gridcolor='black',
                    linecolor='black',
                    mirror=True,
                    showline=True
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                width=600,
                height=500,
                showlegend=False
            )
            plots.append(('Gradient Field', fig_gradient))
        except Exception as e:
            st.info(f"Gradient plot skipped: {e}")
    
    return plots

def save_all_plots_matplotlib(pivot_df, normalized_df, x_label, y_label, colorbar_title, 
                             additional_plots_data, dpi=300, show_values=True):
    """Save all plots using matplotlib"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Main Heatmap
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        
        # Handle NaN values
        plot_data = pivot_df.values.copy()
        if np.isnan(plot_data).any():
            plot_data = np.nan_to_num(plot_data)
        
        # Create heatmap
        im = ax.imshow(plot_data, aspect='auto', cmap='viridis', 
                      extent=[0, len(pivot_df.columns), 0, len(pivot_df.index)])
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_df.columns)) + 0.5)
        ax.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
        ax.set_xticklabels(pivot_df.columns.tolist())
        ax.set_yticklabels(pivot_df.index.tolist())
        
        # Rotate x labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar with vertical orientation
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label(colorbar_title, rotation=270, labelpad=20, fontsize=12, color='black')
        cbar.ax.tick_params(colors='black')
        cbar.ax.yaxis.label.set_color('black')
        
        # Set labels with black color
        ax.set_xlabel(x_label, fontsize=14, color='black')
        ax.set_ylabel(y_label, fontsize=14, color='black')
        ax.set_title('Main Heatmap', fontsize=16, color='black')
        
        # Set axis colors
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        
        # Add grid
        ax.grid(False)
        
        # Add values to cells if requested
        if show_values:
            for i in range(len(pivot_df.index)):
                for j in range(len(pivot_df.columns)):
                    value = pivot_df.values[i, j]
                    if not np.isnan(value):
                        text = ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                                      ha="center", va="center", color="w", fontsize=8)
        
        plt.tight_layout()
        
        # Save to buffer
        heatmap_buffer = io.BytesIO()
        fig.savefig(heatmap_buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr('1_heatmap_main.png', heatmap_buffer.getvalue())
        plt.close(fig)
        
        # 2. Normalized Heatmap (if available)
        if normalized_df is not None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
            
            norm_data = normalized_df.values.copy()
            if np.isnan(norm_data).any():
                norm_data = np.nan_to_num(norm_data)
                
            im = ax.imshow(norm_data, aspect='auto', cmap='viridis', 
                          vmin=0, vmax=1,
                          extent=[0, len(normalized_df.columns), 0, len(normalized_df.index)])
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(normalized_df.columns)) + 0.5)
            ax.set_yticks(np.arange(len(normalized_df.index)) + 0.5)
            ax.set_xticklabels(normalized_df.columns.tolist())
            ax.set_yticklabels(normalized_df.index.tolist())
            
            # Rotate x labels for better visibility
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar with vertical orientation
            cbar = plt.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label(f'{colorbar_title} (normalized)', rotation=270, labelpad=20, fontsize=12, color='black')
            cbar.ax.tick_params(colors='black')
            cbar.ax.yaxis.label.set_color('black')
            
            # Set labels with black color
            ax.set_xlabel(x_label, fontsize=14, color='black')
            ax.set_ylabel(y_label, fontsize=14, color='black')
            ax.set_title('Normalized Heatmap (0-1)', fontsize=16, color='black')
            
            # Set axis colors
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            
            # Add grid
            ax.grid(False)
            
            # Add values to cells if requested
            if show_values:
                for i in range(len(normalized_df.index)):
                    for j in range(len(normalized_df.columns)):
                        value = normalized_df.values[i, j]
                        if not np.isnan(value):
                            text = ax.text(j + 0.5, i + 0.5, f'{value:.3f}',
                                          ha="center", va="center", color="w", fontsize=8)
            
            plt.tight_layout()
            
            # Save to buffer
            norm_buffer = io.BytesIO()
            fig.savefig(norm_buffer, format='png', dpi=dpi, bbox_inches='tight')
            zip_file.writestr('2_heatmap_normalized.png', norm_buffer.getvalue())
            plt.close(fig)
        
        # 3. Contour Plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        
        # Handle NaN values for contour
        contour_data = pivot_df.values.copy()
        if np.isnan(contour_data).any():
            contour_data = np.nan_to_num(contour_data)
        
        # Create meshgrid for contour plot
        X, Y = np.meshgrid(np.arange(len(pivot_df.columns)) + 0.5, 
                          np.arange(len(pivot_df.index)) + 0.5)
        
        # Create contour plot
        contour = ax.contourf(X, Y, contour_data, cmap='viridis', levels=20)
        
        # Add contour lines
        ax.contour(X, Y, contour_data, colors='black', linewidths=0.5, levels=10)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_df.columns)) + 0.5)
        ax.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
        ax.set_xticklabels(pivot_df.columns.tolist())
        ax.set_yticklabels(pivot_df.index.tolist())
        
        # Rotate x labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar with vertical orientation
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
        cbar.set_label(colorbar_title, rotation=270, labelpad=20, fontsize=12, color='black')
        cbar.ax.tick_params(colors='black')
        cbar.ax.yaxis.label.set_color('black')
        
        # Set labels with black color
        ax.set_xlabel(x_label, fontsize=14, color='black')
        ax.set_ylabel(y_label, fontsize=14, color='black')
        ax.set_title('Contour Plot', fontsize=16, color='black')
        
        # Set axis colors
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        
        plt.tight_layout()
        
        # Save to buffer
        contour_buffer = io.BytesIO()
        fig.savefig(contour_buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr('3_contour_plot.png', contour_buffer.getvalue())
        plt.close(fig)
        
        # 4-7. Additional plots from additional_plots_data
        for idx, (plot_name, plot_fig) in enumerate(additional_plots_data):
            # Convert plotly figure to matplotlib for saving
            fig_mpl, ax_mpl = plt.subplots(figsize=(10, 8), dpi=dpi)
            
            if plot_name == 'Density Heatmap':
                # For density heatmap
                data_to_plot = pivot_df.values.copy()
                if np.isnan(data_to_plot).any():
                    data_to_plot = np.nan_to_num(data_to_plot)
                
                im = ax_mpl.imshow(data_to_plot, aspect='auto', cmap='viridis',
                                  extent=[0, len(pivot_df.columns), 0, len(pivot_df.index)])
                
                ax_mpl.set_xticks(np.arange(len(pivot_df.columns)) + 0.5)
                ax_mpl.set_yticks(np.arange(len(pivot_df.index)) + 0.5)
                ax_mpl.set_xticklabels(pivot_df.columns.tolist())
                ax_mpl.set_yticklabels(pivot_df.index.tolist())
                
                plt.setp(ax_mpl.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                cbar = plt.colorbar(im, ax=ax_mpl, orientation='vertical')
                cbar.set_label(colorbar_title, rotation=270, labelpad=20, fontsize=12, color='black')
                cbar.ax.tick_params(colors='black')
                cbar.ax.yaxis.label.set_color('black')
                
                ax_mpl.set_xlabel(x_label, fontsize=14, color='black')
                ax_mpl.set_ylabel(y_label, fontsize=14, color='black')
                ax_mpl.set_title(plot_name, fontsize=16, color='black')
                
            elif plot_name in ['3D Surface', '3D Wireframe']:
                # For 3D plots, create a 2D projection
                data_to_plot = pivot_df.values.copy()
                if np.isnan(data_to_plot).any():
                    data_to_plot = np.nan_to_num(data_to_plot)
                
                # Create a pseudo-3D view using imshow with shading
                im = ax_mpl.imshow(data_to_plot, aspect='auto', cmap='viridis')
                
                ax_mpl.set_xticks(np.arange(len(pivot_df.columns)))
                ax_mpl.set_yticks(np.arange(len(pivot_df.index)))
                ax_mpl.set_xticklabels(pivot_df.columns.tolist())
                ax_mpl.set_yticklabels(pivot_df.index.tolist())
                
                plt.setp(ax_mpl.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                cbar = plt.colorbar(im, ax=ax_mpl, orientation='vertical')
                cbar.set_label(colorbar_title, rotation=270, labelpad=20, fontsize=12, color='black')
                cbar.ax.tick_params(colors='black')
                cbar.ax.yaxis.label.set_color('black')
                
                ax_mpl.set_xlabel(x_label, fontsize=14, color='black')
                ax_mpl.set_ylabel(y_label, fontsize=14, color='black')
                ax_mpl.set_title(f"{plot_name} (2D Projection)", fontsize=16, color='black')
            
            elif plot_name == 'Gradient Field':
                # Skip gradient field for matplotlib export as it's complex
                continue
            
            # Set axis colors
            ax_mpl.spines['bottom'].set_color('black')
            ax_mpl.spines['top'].set_color('black')
            ax_mpl.spines['left'].set_color('black')
            ax_mpl.spines['right'].set_color('black')
            ax_mpl.tick_params(axis='x', colors='black')
            ax_mpl.tick_params(axis='y', colors='black')
            ax_mpl.grid(False)
            
            plt.tight_layout()
            
            # Save to buffer
            plot_buffer = io.BytesIO()
            fig_mpl.savefig(plot_buffer, format='png', dpi=dpi, bbox_inches='tight')
            zip_file.writestr(f'{idx+4}_{plot_name.lower().replace(" ", "_")}.png', plot_buffer.getvalue())
            plt.close(fig_mpl)
        
        # 8. Data table as image
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with formatted values
        table_data = []
        for i in range(len(pivot_df.index)):
            row = []
            for j in range(len(pivot_df.columns)):
                value = pivot_df.values[i, j]
                if np.isnan(value):
                    row.append('NaN')
                else:
                    row.append(f'{value:.3f}')
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        rowLabels=pivot_df.index.tolist(),
                        colLabels=pivot_df.columns.tolist(),
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Set table colors
        for key, cell in table.get_celld().items():
            cell.set_text_props(color='black')
        
        ax.set_title('Data Table', fontsize=16, pad=20, color='black')
        
        plt.tight_layout()
        
        # Save to buffer
        table_buffer = io.BytesIO()
        fig.savefig(table_buffer, format='png', dpi=dpi, bbox_inches='tight')
        zip_file.writestr('8_data_table.png', table_buffer.getvalue())
        plt.close(fig)
    
    return zip_buffer

# Main interface
st.title("🔥 Heatmap Generator for Scientific Publications")
st.markdown("""
Upload data in X,Y,Value format (comma, tab or space separated) or use example data.
Values will be displayed in their original order.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Plot Settings")
    
    # Axis settings
    st.subheader("Axis Settings")
    x_label = st.text_input("X-axis label", value="X")
    y_label = st.text_input("Y-axis label", value="Y")
    colorbar_title = st.text_input("Colorbar title", value="Value")
    
    # Display options
    st.subheader("Display Options")
    sort_numerically = st.checkbox("Sort axes numerically", value=True, 
                                   help="If unchecked, axes will be displayed in the order they appear in the data")
    
    # Font settings
    st.subheader("Font Settings")
    axis_font_size = st.slider("Axis font size", 8, 24, 14)
    tick_font_size = st.slider("Tick font size", 8, 20, 12)
    colorbar_font_size = st.slider("Colorbar font size", 8, 20, 12)
    
    # Display settings
    st.subheader("Data Display")
    show_values = st.checkbox("Show values in cells", value=True)
    value_format = st.selectbox("Value format", 
                               ["Auto", "Integer", "Two decimals", "Three decimals", "Scientific"])
    
    # Color palette selection
    st.subheader("Color Palette")
    
    # Built-in Plotly palettes
    builtin_palettes = [
        "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
        "Greys", "RdBu", "RdYlBu", "Picnic", "Rainbow",
        "Portland", "Jet", "Hot", "Blackbody", "Electric"
    ]
    
    selected_palette = st.selectbox("Select palette", builtin_palettes, index=0)
    
    # Custom palette
    st.markdown("---")
    st.subheader("Custom Palette")
    use_custom_palette = st.checkbox("Use custom palette")
    
    custom_colors = []
    if use_custom_palette:
        color_count = st.slider("Number of colors in palette", 2, 10, 3)
        for i in range(color_count):
            color = st.color_picker(f"Color {i+1}", value="#%06x" % (i * 255 // color_count))
            custom_colors.append(color)
    
    # Contour map settings
    st.markdown("---")
    st.subheader("Contour Map Settings")
    contour_smoothing = st.slider("Smoothing level", 0.0, 3.0, 1.0, 0.1,
                                 help="0 = no smoothing (sharp boundaries), 3 = maximum smoothing")
    show_contour_lines = st.checkbox("Show contour lines", value=True)
    show_contour_labels = st.checkbox("Show contour labels", value=True)
    
    # Additional plots settings
    st.markdown("---")
    st.subheader("Additional Plots")
    show_normalized = st.checkbox("Show normalized plot", value=True)
    show_contour = st.checkbox("Show contour map", value=True)
    show_additional = st.checkbox("Show additional plots", value=True)
    
    # Save settings
    st.markdown("---")
    st.subheader("Save Settings")
    save_dpi = st.slider("DPI for saving", 100, 600, 300)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Data Upload")
    
    # Example data
    example_choice = st.selectbox(
        "Select example data",
        ["Upload your own data", "Example 1: Simple", "Example 2: Temperature data", 
         "Example 3: With gaps", "Example 4: Negative values"]
    )
    
    if example_choice == "Example 1: Simple":
        example_data = """X,Y,Value
A,Jan,10
A,Feb,20
B,Jan,15
B,Feb,25"""
    elif example_choice == "Example 2: Temperature data":
        example_data = """X,Y,Value
Method1,25,0.1
Method1,50,0.2
Method1,100,0.3
Method1,150,0.4
Method1,200,0.5
Method1,250,0.6
Method1,300,0.7
Method1,350,0.8
Method1,400,0.9
Method1,450,1.0
Method1,500,1.1
Method1,550,1.2
Method1,600,1.3
Method1,650,1.4
Method1,700,1.5
Method1,750,1.6
Method1,800,1.7
Method1,850,1.8
Method1,900,1.9
Method1,950,2.0
Method1,1000,2.1
Method2,25,0.15
Method2,50,0.25
Method2,100,0.35
Method2,150,0.45
Method2,200,0.55
Method2,250,0.65
Method2,300,0.75
Method2,350,0.85
Method2,400,0.95
Method2,450,1.05
Method2,500,1.15
Method2,550,1.25
Method2,600,1.35
Method2,650,1.45
Method2,700,1.55
Method2,750,1.65
Method2,800,1.75
Method2,850,1.85
Method2,900,1.95
Method2,950,2.05
Method2,1000,2.15"""
    elif example_choice == "Example 3: With gaps":
        example_data = """A\t1\t0.2
\t2\t0.3
\t3\t0.4
B\t1\t0.25
\t2\t0.35
\t3\t0.45"""
    elif example_choice == "Example 4: Negative values":
        example_data = """X,Y,Value
A,Jan,-10
A,Feb,20
B,Jan,15
B,Feb,-5
C,Jan,30
C,Feb,-15"""
    else:
        example_data = ""
    
    # Data input field
    data_input = st.text_area(
        "Enter data (X, Y, Value separated by comma, tab or space):",
        value=example_data,
        height=200
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Or upload a file",
        type=['txt', 'csv', 'tsv', 'dat']
    )
    
    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')
        data_input = content
    
    # Process button
    if st.button("Generate Heatmaps", type="primary"):
        if data_input.strip():
            with st.spinner("Processing data..."):
                df = parse_data(data_input)
                
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.data_ready = True
                    st.session_state.show_values = show_values
                    st.session_state.sort_numerically = sort_numerically
                    
                    # Store original data for download
                    st.session_state.original_data = data_input
                else:
                    st.error("Failed to process data. Please check the format.")
        else:
            st.warning("Please enter data or upload a file.")

with col2:
    st.header("Data Preview")
    
    if 'df' in st.session_state and st.session_state.get('data_ready', False):
        df = st.session_state.df
        
        st.subheader("Processed Data")
        st.dataframe(df[['X', 'Y', 'Value']], use_container_width=True)
        
        st.subheader("Data Statistics")
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Number of rows", len(df))
            st.metric("Unique X values", df['X'].nunique())
        with col_stats2:
            st.metric("Unique Y values", df['Y'].nunique())
            
            # Safely calculate value range
            try:
                value_min = df['Value'].min()
                value_max = df['Value'].max()
                if pd.isna(value_min) or pd.isna(value_max):
                    st.metric("Value range", "N/A (contains NaN)")
                else:
                    st.metric("Value range", f"{value_min:.2f} - {value_max:.2f}")
            except:
                st.metric("Value range", "N/A")
        
        st.subheader("Pivot Table")
        pivot_df = create_pivot_table(df, sort_numerically)
        if pivot_df is not None:
            st.dataframe(pivot_df, use_container_width=True)
            
            # Show order information
            if len(pivot_df.columns) > 0:
                st.info(f"X-axis order: {', '.join(pivot_df.columns.tolist()[:5])}{'...' if len(pivot_df.columns) > 5 else ''}")
            if len(pivot_df.index) > 0:
                st.info(f"Y-axis order: {', '.join(pivot_df.index.tolist()[:5])}{'...' if len(pivot_df.index) > 5 else ''}")

# Plots area
if 'df' in st.session_state and st.session_state.get('data_ready', False):
    st.markdown("---")
    st.header("Heatmaps")
    
    df = st.session_state.df
    pivot_df = create_pivot_table(df, sort_numerically)
    
    if pivot_df is not None:
        # Value format configuration
        if value_format == "Integer":
            text_format = ".0f"
        elif value_format == "Two decimals":
            text_format = ".2f"
        elif value_format == "Three decimals":
            text_format = ".3f"
        elif value_format == "Scientific":
            text_format = ".2e"
        else:
            # Automatic format selection
            if df['Value'].dtype == np.int64:
                text_format = ".0f"
            else:
                text_format = ".2f"
        
        # Create color scale
        if use_custom_palette and custom_colors:
            # Custom color scale
            colorscale = [[i/(len(custom_colors)-1), color] for i, color in enumerate(custom_colors)]
        else:
            # Use built-in palette
            colorscale = selected_palette
        
        # 1. MAIN HEATMAP
        st.subheader("1. Main Heatmap")
        
        # Prepare data for plotting
        plot_data = pivot_df.values.copy()
        
        # Create text for cells
        if show_values:
            text_matrix = np.empty_like(plot_data, dtype=object)
            for i in range(text_matrix.shape[0]):
                for j in range(text_matrix.shape[1]):
                    value = plot_data[i, j]
                    if np.isnan(value):
                        text_matrix[i, j] = ""
                    else:
                        if text_format == ".0f":
                            text_matrix[i, j] = f"{value:.0f}"
                        elif text_format == ".2f":
                            text_matrix[i, j] = f"{value:.2f}"
                        elif text_format == ".3f":
                            text_matrix[i, j] = f"{value:.3f}"
                        elif text_format == ".2e":
                            text_matrix[i, j] = f"{value:.2e}"
                        else:
                            text_matrix[i, j] = f"{value:.2f}"
        else:
            text_matrix = None
        
        fig1 = go.Figure(data=go.Heatmap(
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
                    font=dict(color='black', size=colorbar_font_size)
                ),
                tickfont=dict(color='black', size=colorbar_font_size-2),
                orientation='v',
                x=1.02,
                xpad=20
            ),
            xgap=1,
            ygap=1
        ))
        
        # Layout configuration for main plot
        fig1.update_layout(
            title=dict(
                text="Heatmap (with borders)",
                font=dict(size=16, color='black'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(
                    text=x_label,
                    font=dict(color='black', size=axis_font_size)
                ),
                tickfont=dict(color='black', size=tick_font_size),
                gridcolor='black',
                linecolor='black',
                mirror=True,
                showline=True,
                zeroline=False,
                showgrid=False,
                type='category',
                tickmode='array',
                tickvals=list(range(len(pivot_df.columns))),
                ticktext=pivot_df.columns.tolist()
            ),
            yaxis=dict(
                title=dict(
                    text=y_label,
                    font=dict(color='black', size=axis_font_size)
                ),
                tickfont=dict(color='black', size=tick_font_size),
                gridcolor='black',
                linecolor='black',
                mirror=True,
                showline=True,
                zeroline=False,
                showgrid=False,
                type='category',
                tickmode='array',
                tickvals=list(range(len(pivot_df.index))),
                ticktext=pivot_df.index.tolist()
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=50, r=100, t=50, b=50)  # Increased right margin for colorbar
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. NORMALIZED HEATMAP (only if all values are non-negative)
        normalized_df = None
        if show_normalized:
            st.subheader("2. Normalized Heatmap (0-1)")
            
            normalized_df = normalize_data(pivot_df)
            
            if normalized_df is not None:
                # Create text for cells
                if show_values:
                    norm_text_matrix = np.empty_like(normalized_df.values, dtype=object)
                    for i in range(norm_text_matrix.shape[0]):
                        for j in range(norm_text_matrix.shape[1]):
                            value = normalized_df.values[i, j]
                            if np.isnan(value):
                                norm_text_matrix[i, j] = ""
                            else:
                                norm_text_matrix[i, j] = f"{value:.3f}"
                else:
                    norm_text_matrix = None
                
                fig2 = go.Figure(data=go.Heatmap(
                    z=normalized_df.values,
                    x=normalized_df.columns.tolist(),
                    y=normalized_df.index.tolist(),
                    colorscale=colorscale,
                    text=norm_text_matrix,
                    texttemplate='%{text}' if norm_text_matrix is not None else None,
                    hoverongaps=False,
                    hoverinfo='x+y+z',
                    colorbar=dict(
                        title=dict(
                            text=f"{colorbar_title} (normalized)",
                            font=dict(color='black', size=colorbar_font_size)
                        ),
                        tickfont=dict(color='black', size=colorbar_font_size-2),
                        orientation='v',
                        x=1.02,
                        xpad=20
                    ),
                    xgap=1,
                    ygap=1
                ))
                
                fig2.update_layout(
                    title=dict(
                        text="Normalized Heatmap",
                        font=dict(size=16, color='black'),
                        x=0.5
                    ),
                    xaxis=dict(
                        title=dict(
                            text=x_label,
                            font=dict(color='black', size=axis_font_size)
                        ),
                        tickfont=dict(color='black', size=tick_font_size),
                        gridcolor='black',
                        linecolor='black',
                        mirror=True,
                        showline=True,
                        zeroline=False,
                        showgrid=False,
                        type='category',
                        tickmode='array',
                        tickvals=list(range(len(normalized_df.columns))),
                        ticktext=normalized_df.columns.tolist()
                    ),
                    yaxis=dict(
                        title=dict(
                            text=y_label,
                            font=dict(color='black', size=axis_font_size)
                        ),
                        tickfont=dict(color='black', size=tick_font_size),
                        gridcolor='black',
                        linecolor='black',
                        mirror=True,
                        showline=True,
                        zeroline=False,
                        showgrid=False,
                        type='category',
                        tickmode='array',
                        tickvals=list(range(len(normalized_df.index))),
                        ticktext=normalized_df.index.tolist()
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    width=800,
                    height=500,
                    margin=dict(l=50, r=100, t=50, b=50)
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Normalization not required or impossible (all values are equal)")
        
        # 3. CONTOUR MAP (smooth transition)
        if show_contour:
            st.subheader("3. Contour Map")
            
            fig3 = create_smooth_contour(pivot_df, selected_palette, contour_smoothing, 
                                        show_contour_lines, show_contour_labels)
            if fig3:
                # Update axis labels
                fig3.update_xaxes(
                    title=dict(
                        text=x_label,
                        font=dict(color='black', size=axis_font_size)
                    )
                )
                fig3.update_yaxes(
                    title=dict(
                        text=y_label,
                        font=dict(color='black', size=axis_font_size)
                    )
                )
                
                # Update colorbar title
                fig3.update_traces(
                    colorbar=dict(
                        title=dict(
                            text=colorbar_title,
                            font=dict(color='black', size=colorbar_font_size)
                        ),
                        tickfont=dict(color='black', size=colorbar_font_size-2),
                        orientation='v',
                        x=1.02,
                        xpad=20
                    )
                )
                
                # Update layout margins for colorbar
                fig3.update_layout(
                    margin=dict(l=50, r=100, t=50, b=50)
                )
                
                st.plotly_chart(fig3, use_container_width=True)
        
        # 4. ADDITIONAL PLOTS
        additional_plots_data = []
        if show_additional:
            st.subheader("4. Additional Visualizations")
            
            additional_plots_data = create_additional_plots(pivot_df, selected_palette, x_label, y_label, colorbar_title)
            
            if additional_plots_data:
                # Display plots in columns
                cols = st.columns(2)
                for idx, (plot_name, plot_fig) in enumerate(additional_plots_data):
                    with cols[idx % 2]:
                        # Update font colors for additional plots
                        plot_fig.update_layout(
                            title_font=dict(color='black'),
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        # Update colorbar orientation and position
                        plot_fig.update_traces(
                            colorbar=dict(
                                orientation='v',
                                x=1.02,
                                xpad=20
                            )
                        )
                        
                        # Update margins for colorbar
                        plot_fig.update_layout(
                            margin=dict(l=50, r=100, t=50, b=50)
                        )
                        
                        # Ensure axes have black labels
                        if 'scene' in plot_fig.layout:
                            # For 3D plots
                            plot_fig.update_scenes(
                                xaxis_title_font=dict(color='black'),
                                yaxis_title_font=dict(color='black'),
                                zaxis_title_font=dict(color='black'),
                                xaxis_tickfont=dict(color='black'),
                                yaxis_tickfont=dict(color='black'),
                                zaxis_tickfont=dict(color='black')
                            )
                        else:
                            # For 2D plots
                            plot_fig.update_xaxes(
                                title_font=dict(color='black'),
                                tickfont=dict(color='black')
                            )
                            plot_fig.update_yaxes(
                                title_font=dict(color='black'),
                                tickfont=dict(color='black')
                            )
                        
                        st.plotly_chart(plot_fig, use_container_width=True)
        
        # Export options
        st.markdown("---")
        st.subheader("Export Options")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📦 Save All Plots (ZIP)"):
                try:
                    with st.spinner("Creating ZIP archive with high-resolution images..."):
                        # Create normalized_df if not already created
                        if normalized_df is None and show_normalized:
                            normalized_df = normalize_data(pivot_df)
                        
                        # Save all plots using matplotlib
                        zip_buffer = save_all_plots_matplotlib(
                            pivot_df, normalized_df, x_label, y_label, colorbar_title,
                            additional_plots_data, dpi=save_dpi, show_values=show_values
                        )
                        
                        # Create download button for ZIP
                        st.download_button(
                            label=f"Download ZIP ({save_dpi} DPI)",
                            data=zip_buffer.getvalue(),
                            file_name=f"heatmap_plots_{save_dpi}dpi.zip",
                            mime="application/zip"
                        )
                        
                        st.success(f"✅ All plots saved with {save_dpi} DPI resolution!")
                        
                except Exception as e:
                    st.error(f"Error saving plots: {str(e)}")
                    
        with col_export2:
            # Export data
            if 'original_data' in st.session_state:
                st.download_button(
                    label="Download Data (CSV)",
                    data=st.session_state.original_data,
                    file_name="heatmap_data.csv",
                    mime="text/csv"
                )
            
        with col_export3:
            # Export pivot table
            pivot_csv = pivot_df.to_csv()
            st.download_button(
                label="Download Pivot Table",
                data=pivot_csv,
                file_name="pivot_table.csv",
                mime="text/csv"
            )

# Data format information
with st.expander("📋 Data Format Information"):
    st.markdown("""
    ### Supported Data Formats:
    
    1. **CSV format**: X,Y,Value separated by comma
    ```
    Temperature,Pressure,Value
    25,1,0.1
    50,1,0.2
    100,1,0.3
    150,1,0.4
    ```
    
    2. **TSV format**: X,Y,Value separated by tab
    ```
    Temperature	Pressure	Value
    25	1	0.1
    50	1	0.2
    100	1	0.3
    150	1	0.4
    ```
    
    3. **Space separated**: X Y Value separated by space
    ```
    Temperature Pressure Value
    25 1 0.1
    50 1 0.2
    100 1 0.3
    150 1 0.4
    ```
    
    ### Key Features:
    
    - **Handles mixed data types**: Supports word-word-number, word-number-number, number-word-number, number-number-number
    - **Preserves axis order**: Values are displayed in the order they appear in your data
    - **Numeric sorting option**: Optional automatic numeric sorting of axis values
    - **High-resolution export**: Save all plots as PNG images with configurable DPI (100-600)
    - **Multiple plot types**: Heatmaps, normalized plots, contour maps, and additional visualizations
    - **Customizable contour smoothing**: Adjust from sharp boundaries to smooth transitions
    - **Proper colorbar placement**: Colorbar titles are now vertical and properly positioned
    """)

# Footer
st.markdown("---")
st.markdown("""
**Heatmap Generator for Scientific Publications** | Optimized for research papers
""")
